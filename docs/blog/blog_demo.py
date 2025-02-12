"""One file for easy reading on the blog."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile

from dotenv import load_dotenv
load_dotenv()

from typing import Any, List, Optional, Union

from datetime import datetime
from dataclasses import dataclass, field
from jinja2 import Template
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

import aiohttp
import pymupdf
import pytesseract
from aiohttp import ClientConnectorError, ClientResponseError, ClientSession
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from requests.exceptions import RequestException
from trafilatura import extract

logging.getLogger().setLevel(logging.INFO)

LLM_NAME = "openai:gpt-4o-mini" # deployment model name
N_RETRIES = 2 # n retries for connection-excerptions during web-search
N_SEARCH_HITS = 10 # how many search results to return from DuckDuckGo
N_PARA_MIN_FOR_REPORT = "three" # minimum number of paragraphs to write in report
N_PARA_MAX_FOR_REPORT = "five"# max number of paragraphs to write in report
N_DOCS_MIN_FOR_REPORT = 3 # min number of documents to download
N_DOCS_MAX_FOR_REPORT = 5 # max number of documents to download

N_PAGE_SUMMARIZE_TRIGGER = 10 # long docs trigger summarization
MAX_PAGES_SUMMARIZE = 50 # max pages to summarize
SNIPPET_LENGTH = int(200*6.7) # ~paragraph returned to main agent

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

class Query(BaseModel):
    """A text query used for web search or QA datasets."""
    text: str = Field(
        description="A natural language question or keyword terms for a query."
    )


class SearchResult(BaseModel):
    """Represents a single search result from search engine."""
    title: str = Field(description="Title of the webpage or online document.")
    url: str = Field(description="URL of the webpage or online document.")
    excerpt: str = Field(
        description="Description of the webpage or online document from the search engine."
    )

    def __str__(self):
        return self.__repr__()


class Doc(BaseModel):
    """Represents a text document, potentially fetched from an online source."""
    title: str = Field(description="Title of the document.")
    url: Optional[str] = Field(
        default="",
        description="Optional URL of the webpage or online document (if fetched online)."
    )
    text: str = Field(description="Text content of the document.")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Doc):
            return False
        return self.title == other.title and self.url == other.url and self.text == other.text    


class AskClarifyingQuestionOfUser(BaseModel):
    """Represents the agent's question for user to clarify task."""
    questions: str = Field(
        description="Questions to ask the user to help clarify their goals and resolve their research intent"
    )


class Footnote(BaseModel):
    """Represents a footnote in the research report."""
    id:int = Field(description="Footnote number to map [NUMBER] in the body of the research report to this footnote.")
    description:str = Field(description="Brief description of the note or reference.")
    url:str = Field(description="URL of document being referenced.")
    
    def __str__(self):
        return (
            f"[{self.id}]: "
            f"{self.description} "
            f"[link]({self.url})"
        )
    
class ResearchReport(BaseModel):
    """represents the final result of research to present to the user."""
    title:str
    body:str
    footnotes:List[Union[str,Footnote]]
    
    def to_markdown(self):
        out = f"# {self.title}\n {self.body}"
        if self.footnotes:
            footnotes = "\n".join([str(footnote) for footnote in self.footnotes])
            return (out + "\n\nNotes:\n" + footnotes)
            
        return out
    
    def __str__(self):
        return self.to_markdown()


class WarningTooFewDocs(BaseModel):
    """Warns the Research Assistant that there aren't enough documents to write a report."""
    user_intent_long:str= Field(
        description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
    )
    n_docs:int=0
    warning:Optional[str]=(
        "Not enough documents downloaded to Knowledge Base to write report. Please fetch more documents."
    )
    
    def model_post_init(self, *args, **kwargs):
        """Crafts a message to return to Research Assistant."""
        self.warning = (
            f"There are only {self.n_docs} documents downloaded to the knowledge base. Please conduct some "
            "more web-searches (via `web_search`) and/or download more relevant documents (via `fetch_online_doc`) "
            f"so that I have enough documents to write a report about: '{self.user_intent_long}'."
        )
    def __str__(self):
        return self.warning
    
    def __repr__(self):
        return f"WarningTooFewDocs(warning='{self.warning}')"
    

class SearchIntentResult(BaseModel):
    """Represents the agent's answer in which agent has resolved a user's search intent."""
    user_intent_short: str = Field(
        description="A succinct summary of the user's search intent."
    )
    user_intent_long: str = Field(
        description="A detailed outline of the user's intent, topic, desired outputs, and relevant entities."
    )
    recommended_queries: Optional[List[Query]] = Field(
        description="Optional list of helpful queries for searching the web to satisfy user intent."
    )

# -- Dependencies for agents-- 

@dataclass
class ResearchAssistantDeps:
    """Dependencies for the ResearchAssistantAgent."""
    docs: List[Doc] = field(default_factory=list)  # Use default_factory=list
    disambiguation_agent: Optional[Agent] = None # slot for agent help with disambiguation
    report_writer_agent: Optional[Agent] = None # slot for report writer agent
    summarizer_agent: Optional[Agent] = None # slot for report writer agent

@dataclass
class DisambiguationAgentDeps:
    """Dependencies for the Search Intent Agent."""
    pass

@dataclass
class SummarizerAgentDeps:
    """Dependencies for the Summarizer."""
    pass

@dataclass
class ReportWriterDeps:
    """Dependencies for the ReportWriter Agent."""
    user_intent_long: str = Field(
        description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
    )
    docs: List[Doc] = field(default_factory=list)


## -- Initialize Agents --

# Agent: disambiguates user search intent
search_intent_agent = Agent(
    LLM_NAME,  # Or your preferred model
    deps_type=DisambiguationAgentDeps,
    result_type=Union[AskClarifyingQuestionOfUser, SearchIntentResult], # expected output
    system_prompt= """You are a research assistant tasked with helping a user articulate their research goals. You must try to clarify the user's research intent and resolve any ambiguous entities. Based on your description of the user's intent (`user_intent_short` and `user_intent_long`), other researchers will run queries and write a research report for the user. You must guide their research.
        ## AVAILABLE TOOLS
        You may simply chat with the user to clarify their intent and topic, or, optionally, you may use some available tools, including:
	i) `web_search` - to search DuckDuckGo and get back search-results about a topic relevant to the user; or
        ii) `fetch_online_doc` - to fetches a website's content. For example, this could be useful to fetch a wikipedia about one of the user's topics or entities.
        ## Output Format
        If you must ask the user a question to disambiguate any entities and clarify their research goals, return an `AskClarifyingQuestionOfUser` object. Or, if you've successfully resolved the intent of the user, return a `SearchIntentResult` that includes `user_intent_short` and `user_intent_long`.""",
    result_retries=2,
)


# Agent: summarizes big docs
summarizer_agent = Agent(
    LLM_NAME, 
    deps_type=SummarizerAgentDeps,
    result_type=str,
    system_prompt=(
        "You are tasked with summarizing long documents into one or two pages.\n"
        "First chunk of the document will be retained as content -- you do not need to summarize the first chunk.\n"
        "In your summary of all the content after the first chunk try to be as extractive as possible, which is to say, include as much exact content and key-excerpts as possible. The extract excerpts should balance good coverage of all the content as well as the most important content.\n"
        "Your one-to-two page extractive summary will be used for drafting a research report "
        "so exactitude and faithful excerpts are important."
    ),
    result_retries=2,
)


# Agent: writes the final report
report_writer_agent = Agent(
    LLM_NAME,
    deps_type=ReportWriterDeps,
    result_type=Union[ResearchReport, WarningTooFewDocs],
    system_prompt=(
        "You are a technical writer. You will receive a collection of documents and your task is to synthesize their content into a concise research report. "
        "Do not invent any information or introduce any external knowledge, only use the provided documents as the basis for your report. "
        "The report should be well-structured, objective and aim to provide a comprehensive overview of the topic based on the gathered information. "
        "Ensure to properly cite the source for each piece of information using footnotes (use the `footenote` attribute of the ResearchReport."
    ),
    result_retries=2,
)

@report_writer_agent.system_prompt
async def add_documents_to_agent_prompt(ctx: RunContext[ReportWriterDeps]) -> str:
    """Add documents from dependencies to the system prompt."""
    docs_template = Template("""## DOCUMENTS:
        Please base your report on the following documents:
        <documents>
        {% for doc in docs %}
            <document>
                <h1>{{doc.title}}</h1>
                <url>{{doc.url}}</url>
                <body>{{doc.text}}</body>
            </document>
        {% endfor %}
        </documents>
        Do not make-up or invent any facts outside of the contents of these documents."""
    )
    docs_xml = docs_template.render(docs=ctx.deps.docs)
    return docs_xml


# Main Agent: delegates tasks to other agents
research_assistant_agent = Agent(
    LLM_NAME,
    deps_type=ResearchAssistantDeps,
    result_type=Union[AskClarifyingQuestionOfUser, ResearchReport],
    system_prompt=(
        f"You are a research assistant with access to tools for web searches, document "
	"retrieval, and report synthesis.\n"
	"- Begin by using the `clarify_intent` tool to refine the user's goals.\n"
	"- Search the web with the `web_search` tool, finding the urls of relevant docs.\n"
	"- Given search results, fetch relevant documents using the `fetch_online_doc` "
	" tool, which downloads documents to your Knowledge Base.\n"
	f"- Ensure you download {N_DOCS_MIN_FOR_REPORT}-to-{N_DOCS_MAX_FOR_REPORT} documents. "
        "Use the `n_docs_downloaded` tool to periodically check how many documents you downloaded."
	f"finally, compile a {N_PARA_MIN_FOR_REPORT}-to-{N_PARA_MAX_FOR_REPORT} "
	"paragraph report with the `write_report` tool."
    )
)


@report_writer_agent.system_prompt
@search_intent_agent.system_prompt
@summarizer_agent.system_prompt
@research_assistant_agent.system_prompt
async def add_current_date(ctx: RunContext[ReportWriterDeps]) -> str:
    """Add today's date to agent's system to ensure focus on latest info."""
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"
    

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
    run_result = await summarizer_agent.run(query,deps=SummarizerAgentDeps())
    return first_chunk + "\n" + str(run_result.data)


# helper function to add a document to knowledge base
async def add_doc(deps:ResearchAssistantDeps, doc: Doc) -> str:
    """Adds a document to the agent's document cache."""
    if not doc.text:
        # noting to add to knowledge base
        return f"Document {doc.title} - {doc.url} couldn't be retrieved, ignoring it."
    
    n_char_threshold = int(N_PAGE_SUMMARIZE_TRIGGER*500*6.7) # 500 words per page * n-char per word
    
    if len(doc.text) > n_char_threshold:
        # if document is too large, summarize...
        doc.text = await summarize_doc(doc, n_char_threshold, deps.summarizer_agent)
    
    # add document to knowledge base
    deps.docs.append(doc)
    
    # return summary message to Agent
    return (
        f"Downloaded and added {doc.title} ({doc.url if doc.url!=doc.title else ''}). "
        f"Snippet:'{doc.text[:SNIPPET_LENGTH]}...'\n"
        f"There are now {len(deps.docs)} in knowledge base."
    )


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
            user_intent_long=user_intent_long, n_docs=len(ctx.deps.docs)
        )
            
    # call research writer agent to draft final report
    query_writer = (
        f"Here is what the report should be about: '{user_intent_long}' \n"
        f"Please write a report approximately {N_PARA_MIN_FOR_REPORT} to {N_PARA_MAX_FOR_REPORT} "
        "paragraphs long, that satisfies the user's topic and interest."
    )
    report = await ctx.deps.report_writer_agent.run(query_writer,deps=report_writer_deps)
    return report


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
            logging.info(f" - got {len(results_raw)} results for {query.text}")
            return results
        except DuckDuckGoSearchException as e:
            if "Ratelimit" in str(e) and attempt < n_retries:
                logging.warning(f"Rate limit hit, retrying in 5 seconds... Attempt {attempt + 1}/{n_retries}")
                await asyncio.sleep(4*(1+attempt)) # backoff
            else:
                raise  # Re-raise the exception if it's not a rate limit error or all retries hav


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
    
    search_results = await _web_search(query)
    return search_results


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
        raise  # Re-raise to be handled by the caller
    except Exception as e:
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
                logging.warning(f"All retries failed: returning excerpt only")
                break
        
        except Exception as e:
            logging.warning(f"Non-retryable error occurred: {e}")
            break
    
    return ""

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
    
    doc = Doc(title=title, url=url, text=doc_content)
    # check already in knowledge base
    if doc  in ctx.deps.docs:
        return f"Document {doc.title} ({doc.url}) is already in Knowledge Base."
    
    # add doc to ctx/internal knowledge base (or excerpt, if failed)
    msg = await add_doc(ctx.deps, doc)
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
            "No documents downloaded. Please do some web-searches and fetch "
            "some online documents/webpages as preparation for writing the report."
        )
    if n_docs > 0:
        titles = ", ".join([doc.title for doc in ctx.deps.docs])
    if n_docs == 1:
        return (
            f"There is only 1 document downloaded. Its title is {titles}. Please "
            "download some more online documents/webpages to support the research request."
        )
    return (
        f"There have been {n_docs} documents/webpages downloaded and whose text "
	f"has been extracted. Their titles are {titles}"
    )





## -- Example usage --

# initialize the deps and attach sub-agents
deps = ResearchAssistantDeps(
    docs=[],
    disambiguation_agent=search_intent_agent,
    report_writer_agent=report_writer_agent,
    summarizer_agent=summarizer_agent
)

# first chat
out = research_assistant_agent.run_sync(
    "I'd like a small report for investors about Thomson Reuters (TSX:TRI), such as: who are its competitors, what are analysts saying about it's growth potential.", deps=deps
)
print(out.data)
## AskClarifyingQuestionOfUser(questions="Could you specify what aspects of Thomson Reuters' competitors' growth potential you are most interested in? For example, are you looking for financial growth metrics, market share analysis, or strategic initiatives? Additionally, do you have specific competitors in mind that you would like to focus on?")


# 2nd chat: follow-up with more clarifying details
out = research_assistant_agent.run_sync(
    "I'm specifically interested in Thomson Reuter's potential for organic revenue growth, as compared to it's main competitors in the legal and tax services domains. Please first figure out its main legal and tax competitors before conducting research.",
    deps=deps,
    message_history=out.new_messages()) # ADD MESSAGE HISTORY to continue chat
)
print(out.data)
