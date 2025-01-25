from __future__ import annotations

import asyncio
import logging
import os

from typing import List, Optional, Union

from datetime import datetime
from jinja2 import Template
from pydantic import Field
from pydantic_ai import Agent, RunContext

from src.models import Doc, Query, SearchResult
from src.utils.webtools import _fetch_online_doc, _web_search
from src.usecase.company_research.config import (
    LLM_NAME, N_DOCS_MIN_FOR_REPORT, N_PARA_MIN_FOR_REPORT,N_PARA_MAX_FOR_REPORT
)
from src.usecase.company_research.prompts import (
    systemprompt_search_agent,
    systemprompt_summarizer,
    systemprompt_writer,
    systemprompt_researcher,
    prompt_document_xml
)
from src.usecase.company_research.models import (
    AskClarifyingQuestionOfUser,
    Footnote,
    ResearchReport,
    SearchIntentResult,
    WarningTooFewDocs,
    ResearchAssistantDeps,
    DisambiguationAgentDeps,
    SummarizerAgentDeps,
    ReportWriterDeps,
)
from src.usecase.company_research.utils import add_doc, summarize_doc

logging.getLogger().setLevel(logging.INFO)


## -- Initialize Agents --

# Agent: disambiguates user search intent
search_intent_agent = Agent(
    LLM_NAME, 
    deps_type=DisambiguationAgentDeps,
    result_type=Union[AskClarifyingQuestionOfUser, SearchIntentResult], # expected output schema
    system_prompt=systemprompt_search_agent, 
    result_retries=2,
)

# Agent: summarizes big docs
summarizer_agent = Agent(
    LLM_NAME,  # Or your preferred model
    deps_type=SummarizerAgentDeps,
    result_type=str,
    system_prompt=systemprompt_summarizer,
    result_retries=2,
)

# Agent: writes the final report
report_writer_agent = Agent(
    LLM_NAME,  # Or your preferred model
    deps_type=ReportWriterDeps,
    result_type=Union[ResearchReport, WarningTooFewDocs],
    system_prompt=systemprompt_writer,
    result_retries=2,
)

# Main Agent: delegates tasks to other agents
research_assistant_agent = Agent(
    LLM_NAME,  # Or your preferred model
    deps_type=ResearchAssistantDeps,
    result_type=Union[AskClarifyingQuestionOfUser, ResearchReport],
    system_prompt=systemprompt_researcher,
)


## -- dynamic additions to the system prompts
@report_writer_agent.system_prompt
async def add_documents_to_agent_prompt(ctx: RunContext[ReportWriterDeps]) -> str:
    """Add documents from dependencies to the system prompt."""
    docs_template = Template(prompt_document_xml)
    docs_xml = docs_template.render(docs=ctx.deps.docs)
    return docs_xml

@report_writer_agent.system_prompt
@search_intent_agent.system_prompt
@summarizer_agent.system_prompt
@research_assistant_agent.system_prompt
async def add_current_date(ctx: RunContext[ReportWriterDeps]) -> str:
    """Add today's date to agent's system to ensure focus on latest info."""
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"


## -- Tools for Agents --

@search_intent_agent.tool
@research_assistant_agent.tool
async def web_search(
        ctx: RunContext[DisambiguationAgentDeps|ResearchAssistantDeps],
        query: Query | str
) -> List[SearchResult]:
    """Performs a web search and returns a list of search results.

    This tool hits a search engine api using the provided query and
    returns a list of 'SearchResult'. Each result includes a title, URL,
    and a short excerpt from the search engine.

    Args:
        ctx: The RunContext containing agent dependencies.
        query: Search query or a `Query(text="[search query")` object

    Returns:
        List[SearchResult]: A list of 'SearchResult' objects representing 
                           the web search results (url, title, excerpt).
    """    
    if isinstance(query, str):
        query = Query(text=query)
    
    logging.info("Beginning duckduckgo search about `%s`" % query.text)
    search_results = await _web_search(query)
    return search_results


@search_intent_agent.tool            
@research_assistant_agent.tool
async def fetch_online_doc(
        ctx: RunContext[ResearchAssistantDeps|DisambiguationAgentDeps],
        url: Union[SearchResult,str],
        title: Optional[str] = None,
        excerpt: Optional[str] = None,
) -> str:
    """Fetches and stores content from a given URL.

    This tool attempts to download the contents of an online document document
    from a provided URL, supporting both HTML and PDFs. If successful, the
    extracted contents are stored as a 'Doc' object in the knowledge base.
    
    Args:
        ctx: The RunContext containing agent dependencies.
        url: The URL of the document to fetch, either as a string or a 
             'SearchResult' returned from the `web_search` tool.
        title: Optional title accompanying url, as provided from a
             'SearchResult' result returned by the `web_search` tool.
        excerpt: Optional text snippet accompanyiong the url, as provided
             by 'SearchResult' result returned by the `web_search` tool.
    
    Returns:
        str: A message indicating success or failure of fetching document.
    """    
    # ensure deps and url are both provided, else send message back to agent
    if (ctx.deps is None or url is None):
        logging.warning(f'WARNING: `fetch_online_doc` passed on None ctx or url ({url})')
        return (
            f"Error: `fetch_online_doc` requires argument `RunContext[ResearchAssistantDeps|"
            "DisambiguationAgentDeps]` and `url: Union[SearchResult,str]`. Please try callling"
            "`fetch_online_doc` agains with `ctx:RunContext[ResearchAssistantDeps|"
            "DisambiguationAgentDeps]` specified and a proper url"
            f"`{str(url) if url is not None else ''}`."
        )

    # Agent may pass us a str-url or SearchResult object
    if isinstance(url, SearchResult):
        title = url.title
        excerpt = url.excerpt
        url = url.url
    
    if not title:
        title = url
    
    # download text of online document
    doc_content = await _fetch_online_doc(url)

    # fallback to excerpt if fetch failed
    doc_content = doc_content if doc_content else excerpt # fallback
    
    # add doc to ctx/internal knowledge base (or excerpt, if failed)
    msg = await add_doc(ctx.deps, Doc(title=title, url=url, text=doc_content))
    
    # return message back to Agent about successful addition of document
    return msg


@research_assistant_agent.tool
async def n_docs_downloaded(ctx: RunContext[ResearchAssistantDeps]) -> str:
    """Number of downloaded documents in internal knowledge base.

    This tool returns the number of documents stored in the internal knowledge
    base. If there are too few documents, it advises that more research should
    be conducted to gather more documents.

    Args:
        ctx: The RunContext containing agent dependencies.

    Returns:
        str: A message conveying the number of downloaded documents and 
             potentially suggesting further document fetching.
    """    
    n_docs = len(ctx.deps.docs)
    if n_docs == 0:
        # warn agent about not-enough documents downloaded
        return (
            "No webpages or online-documents have been downloaded. Please do some web-searches "
            "and fetch some online documents/webpages as preparation for writing the report of "
            "the user."
        )
    if n_docs > 0:
        titles = ", ".join([doc.title for doc in ctx.deps.docs])
    if n_docs == 1:
        return (
            f"There is 1 webpages/online-documents downloaded. Its title is {titles}. Please "
            "download some more online documents/webpages to support the research request prior "
            "to writing the report for the user."
        )
    # Inform agent about the number of downloaded documents
    return (
        f"There have been {n_docs} documents/webpages downloaded and whose text has been extracted. "
        "Their titles are {titles}"
    )


@research_assistant_agent.tool
async def get_all_docs(ctx: RunContext[ResearchAssistantDeps]) -> List[Doc]:
    """Retrieves all downloaded documents from the knowledge base.

    Args:
        ctx: The RunContext containing agent dependencies.

    Returns:
        List[Doc]: A list of 'Doc' objects representing all documents 
                   stored in the internal knowledge base.
    """
    return ctx.deps.docs


@research_assistant_agent.tool
async def clarify_intent(
    ctx: RunContext[ResearchAssistantDeps], query: str
) -> Union[AskClarifyingQuestionOfUser, SearchIntentResult]:
    """Clarifies research intent, returning an expanded description or clarifying questions.

    This tool attempts to resolve the user's search intent based on the provided query. 
    It may return a set of clarifying questions to ask the user and refine the search,
    or a concrete representation of the resolved search intent.

    Args:
        ctx: The RunContext containing agent dependencies.
        query: The user's initial request as `Query(text=...)`

    Returns:
        Union[AskClarifyingQuestionOfUser, SearchIntentResult]: 
            Either an `AskClarifyingQuestionOfUser` object containing questions for the user 
            to resolve ambiguities, or a `SearchIntentResult` object with `user_intent_short`,
            `user_intent_long` and `recommended_queries` for subsequent web-search.
    """
    return await ctx.deps.disambiguation_agent.run(query)


@research_assistant_agent.tool
async def write_report(
        ctx: RunContext[ResearchAssistantDeps],
        user_intent_long:str = Field(
            description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
        ),
        user_intent_short:Optional[str]=None
) -> Union[ResearchReport, WarningTooFewDocs]:
    """Generates a research report based on the user's intent and available documents.

    This tool attempts to create a comprehensive research report addressing the 
    user's specified intent. It leverages the documents stored in the knowledge base
    to synthesize the information. If there are insufficient documents to produce a
    meaningful report, a warning is returned.

    Args:
        ctx: The RunContext containing agent dependencies.
        user_intent_long: A detailed description of the user's research intent.
        user_intent_short: An optional shorter version of the user's intent. 

    Returns:
        Union[ResearchReport, WarningTooFewDocs]: 
            Either a 'ResearchReport' object containing the generated report 
            or a 'WarningTooFewDocs' object if more documents are needed.
    """    
    
    if (not user_intent_long) and user_intent_short:
        user_intent_long = user_intent_short
    assert user_intent_long
    
    # add the docs to the report writer's dependencies/context
    report_writer_deps = ReportWriterDeps(
        user_intent_long = user_intent_long,
        docs = ctx.deps.docs # get downloaded docs from knowledge base 
    )
    if len(report_writer_deps.docs) < N_DOCS_MIN_FOR_REPORT:
        # warng Main agent to fetch more documents
        return WarningTooFewDocs(
            user_intent_long = user_intent_long,
            n_docs = len(report_writer_deps.docs)
        )
    
    logging.info(f"I will write a report on '{user_intent_long}' using {len(report_writer_deps.docs)} docs.")
            
    # call research writer agent to draft final report
    query_writer = (
        "## GOAL OF REPORT: \n"
        f"Here is what the report should be about: '{user_intent_long}' \n"
        f"Please write a report approximately {N_PARA_MIN_FOR_REPORT} to {N_PARA_MAX_FOR_REPORT} "
        "paragraphs long, that satisfies the user's topic and interest."
    )
    report = await ctx.deps.report_writer_agent.run(query_writer,deps=report_writer_deps)
    return report

