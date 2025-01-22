from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from typing import List, Optional, Union

from jinja2 import Template
from pydantic_ai import Agent, RunContext

from src.config import N_PAGE_SUMMARIZE_TRIGGER, LLM_NAME
from src.models import Doc, Query, SearchResult
from src.utils.webtools import _fetch_online_doc, _web_search
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
    SummarizerAgentDeps
    ReportWriterDeps,
)
from src.usecase.company_research.utils import add_doc, summarize_doc

logging.getLogger().setLevel(logging.INFO)

# load llm credentials
load_dotenv()


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


## -- Tools for Agents --

@search_intent_agent.tool
@research_assistant_agent.tool
async def web_search(
        ctx: RunContext[DisambiguationAgentDeps|ResearchAssistantDeps],
        query: Query | str
) -> List[SearchResult]:
    """Performs a web search using DuckDuckGo and returns the results."""
    
    if isinstance(query, str):
        query = Query(text=query)
    logging.info("Beginning duckduckgo search about `%s`" % query.text)
    
    n_retries = 2
    for attempt in range(n_retries + 1):
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


@search_intent_agent.tool            
@research_assistant_agent.tool
async def fetch_online_doc(
        ctx: RunContext[ResearchAssistantDeps|DisambiguationAgentDeps],
        url: Union[SearchResult,str],
        title: Optional[str] = None,
        excerpt: Optional[str] = None,
) -> str:
    """Fetches an online document and extracts its text content (both PDF and HTML)."""

    # ensure deps and url are both provided, else send message back to agent
    if (ctx.deps is None or url is None):
        logging.warning(f'WARNING: `fetch_online_doc` passed on None ctx or url ({url})')
        return (
            f"Error: `fetch_online_doc` requires argument `RunContext[ResearchAssistantDeps|"
            "DisambiguationAgentDeps]` and `url: Union[SearchResult,str]`. Please try callling"
            "`fetch_online_doc` agains with `ctx:RunContext[ResearchAssistantDeps|"
            "DisambiguationAgentDeps]` specified and a proper argument for url"
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
    
    # add doc to ctx/internal knowledge base (or excerpt, if failed)
    msg = await add_doc(ctx.deps, Doc(title=title, url=url, text=doc_content))
    
    # return message back to Agent about successful addition of document
    return msg


@research_assistant_agent.tool
async def n_docs_downloaded(ctx: RunContext[ResearchAssistantDeps]) -> str:
    """Returns the number of documents stored in the agent's memory."""
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
    """Retrieves all documents from the agent's memory."""
    return ctx.deps.docs


@research_assistant_agent.tool
async def clarify_intent(
    ctx: RunContext[ResearchAssistantDeps], query: str
) -> Union[AskClarifyingQuestionOfUser, SearchIntentResult]:
    """This tool will try to understand the user's intent, potentially asking clarifying questions. """
    return await ctx.deps.disambiguation_agent.run(query)


@report_writer_agent.system_prompt
async def add_documents_to_agent_prompt(ctx: RunContext[ReportWriterDeps]) -> str:
    # DocString: Add documents from dependencies to the system prompt.
    docs_template = Template(prompt_document_xml)
    docs_xml = docs_template.render(docs=ctx.deps.docs)
    return docs_xml

@research_assistant_agent.tool
async def write_report(
        ctx: RunContext[ResearchAssistantDeps],
        user_intent_long:str = Field(
            description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
        ),
        user_intent_short:Optional[str]=None
) -> Union[ResearchReport, WarningTooFewDocs]:
    """Calls ReportWriter Agent to synthesize the downloaded documents into a report."""
    if (not user_intent_long) and user_intent_short:
        user_intent_long = user_intent_short
    assert user_intent_long
    # add the docs to the report writer's dependencies/context
    report_writer_deps = ReportWriterDeps(
        user_intent_long = user_intent_long,
        docs = ctx.deps.docs # get downloaded docs from 
    )
    if len(report_writer_deps.docs)<N_DOCS_MIN_FOR_REPORT:
        return WarningTooFewDocs(
            text = (
                f"There are only {len(report_writer_deps.docs)} documents downloaded to "
                "the knowledge base. Please conduct some more web-searches (`web_search`) and/or "
                "download more relevant documents (`fetch_online_doc`) so that I can properly "
                f" write a report about: '{user_intent_long}'."
            )
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

