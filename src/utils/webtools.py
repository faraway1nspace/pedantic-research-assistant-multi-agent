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


async def _fetch_pdf_content(url: str) -> str:
    """Fetches and extracts text content from a PDF URL."""
    doc_content = ""
    connector = aiohttp.TCPConnector()
    try:
        async with aiohttp.ClientSession(
            connector=connector, max_line_size=8190 * 2, max_field_size=8190 * 2
        ) as session:
            async with session.get(url, headers=HEADERS) as response:
                response.raise_for_status()
                content = await response.read()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        pdfdoc = pymupdf.open(temp_file_path)
        for pdfpage in pdfdoc:
            doc_content += pdfpage.get_text()
    except (ClientResponseError, ClientConnectorError) as e:
        logging.warning(f"Error fetching PDF from {url}: {e}")
        raise  # Re-raise to be handled by the caller
    except Exception as e:
        logging.warning(f"Error processing PDF from {url}: {e}")
        raise  # Re-raise to be handled by the caller            
    return doc_content


async def _fetch_html_content(url: str) -> str:
    """Fetches and extracts text content from an HTML URL."""
    doc_content = ""
    connector = aiohttp.TCPConnector()
    try:
        # session with larger headers than default
        async with aiohttp.ClientSession(
            connector=connector, max_line_size=8190 * 2, max_field_size=8190 * 2
        ) as session:    
            async with session.get(url, headers=HEADERS) as response:
                response.raise_for_status()
                doc_html = await response.text()
        
        doc_content = extract(doc_html, url=url)
        if not doc_content:
            doc_content = html_quick_clean(doc_html)
    except RequestException as e:
        logging.warning(f"Error fetching or processing HTML: {e}")
        raise  # Re-raise to be handled by the caller
    return doc_content


async def _fetch_online_doc(url:str)->str:
    """Fetches an online document and extracts its text content (both PDF and HTML)."""
    
    logging.info("Attempting fetch: `%s`" % url)

    # n_retries in case of connectivity exceptions
    for attempt in range(N_RETRIES + 1): 
        try:
            if url.lower().endswith(".pdf"):
                doc_content = await _fetch_pdf_content(url)
            else:
                doc_content = await _fetch_html_content(url)
            return doc_content
        
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
    
    return ""
