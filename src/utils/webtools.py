"""Methods for searching DuckDuckGo and fetching websites."""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from typing import List, Optional, Union

import aiohttp
import pymupdf
import pytesseract
from aiohttp import ClientConnectorError, ClientResponseError, ClientSession
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from trafilatura import extract

from src.models import SearchResult
from src.config import N_SEARCH_HITS, N_RETRIES


logging.getLogger().setLevel(logging.INFO)


async def _web_search(query: Query | str) -> List[SearchResult]:
    """Performs a web search using DuckDuckGo and returns the results."""

    # loop over possible excerptions raised (connectivity issues
    for attempt in range(N_RETRIES + 1):
        try: 
            results_raw = await asyncio.to_thread(DDGS().text, query.text, max_results=N_SEARCH_HITS)
            results: List[SearchResult] = []
            for result_raw in results_raw:
                results.append(
                    SearchResult(
                        title=result_raw["title"],
                        url=result_raw["href"],
                        excerpt=result_raw["body"],
                    )
                )
            logging.info(f" - got {len(results_raw)} results")
            return results
        except DuckDuckGoSearchException as e:
            if "Ratelimit" in str(e) and attempt < n_retries:
                logging.warning(f"Rate limit hit, retrying in 5 seconds... Attempt {attempt + 1}/{n_retries}")
                await asyncio.sleep(4*(1+attempt)) # backoff
            else:
                raise  # Re-raise the exception if it's not a rate limit error or all retries hav


def html_quick_clean(html:str)->str:
    """Crude removal of html tags and javascript."""
    text = re.sub(r"\<script\>(.*?)\<\/script\>","",html,flags=re.DOTALL|re.MULTILINE)
    return re.sub(r"\<(.*?)\>","",text,flags=re.DOTALL|re.MULTILINE)


async def _fetch_online_doc(url:str)->str:
    """Fetches an online document and extracts its text content (both PDF and HTML)."""
    
    doc_content = excerpt if excerpt else "" # fallback
    logging.info("Attempting fetch: `%s`" % url)

    # n_retries in case of connectivity exceptions
    for attempt in range(N_RETRIES + 1): 
        try:
            if url.lower().endswith(".pdf"):
                doc_content = await _fetch_pdf_content(url)
            else:
                doc_content = await _fetch_html_content(url)
            break  # Exit the loop if successful
        
        except (ClientResponseError, ClientConnectorError) as e:
            logging.info(f"Attempt {attempt + 1} failed: {e}")
            if attempt < N_RETRIES:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.warning(f"All retries failed: returning excerpt only: '{doc_content}'")
                break
        
        except Exception as e:
            logging.warning(f"Non-retryable error occurred: {e}")
            break
    
    return doc_content
