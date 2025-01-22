"""Helper functions used inside Agent tools"""
import asyncio
import logging

from pydantic_ai import Agent

from src.models import Doc
from src.config import N_PAGE_SUMMARIZE_TRIGGER
from src.usecase.company_research.models import (
    ResearchAssistantDeps,
    SummarizerAgentDeps
)

logging.getLogger().setLevel(logging.INFO)

# controls the truncation of document-text shown to main-agent when downloading documents
SNIPPET_LENGTH = 300*6.7

# total number of pages to summarize; beyond which are ignored
MAX_PAGES_SUMMARIZE = 50


async def summarize_doc(doc:Doc, n_char_threshold:int, summarizer_agent:Agent) -> str:
    """Summarizes a document if it is too long."""
    first_chunk = doc.text[:n_char_threshold] # exclude from summary
    text_to_summarize = doc.text[n_char_threshold:] # to summarize
    max_pages_in_char = int(MAX_PAGES_SUMMARIZE*500*6.7) # ~50 pages (in characters)
    if len(text_to_summarize) > max_pages_in_char:
        text_to_summarize = text_to_summarize[:max_pages_in_char]
    
    # make summarization query
    query = (
        f"## Title: {doc.title}\n"
        f"##First Chunk (not required in summary, for use as context):\n{first_chunk}\n\n"
        "## BEGIN SUMMARIZATION\n"
        "Please do a one-to-two page extractive summary of the remainder of the document:\n"
        f"{text_to_summarize}"
    )
    # use agent to summarize long doc
    summary_second_chunk = await summarizer_agent.run(query,deps=SummarizerAgentDeps())

    # return chunk and summarized portion to main agent
    return first_chunk + "\n" + summary_second_chunk


async def add_doc(deps:ResearchAssistantDeps, doc: Doc) -> str:
    """Adds a document to the agent's document cache."""
    if not doc.text:
        # noting to add to knowledge base
        return f"Document {doc.title} - {doc.url} couldn't be retrieved, ignoring it."
    
    n_char_threshold = int(N_PAGE_SUMMARIZE_TRIGGER*500*6.7) # 500 words per page * n-char per word
    
    if len(doc.text) > n_char_threshold:
        # if document is too large, summarize...
        logging.info('document too long: summarizing for knowledge base')
        doc.text = await summarize_doc(doc, n_char_threshold, deps.summarizer_agent)
    
    # add document to knowledge base
    deps.docs.append(doc)
    logging.info(f"Added document '{doc.title}' to document cache.")        
    
    # return summary message to Agent
    return f"Downloaded and added '{doc.url}' to knowledge base:\nSnippet:'{doc.text[:SNIPPET_LENGTH]}...'"
