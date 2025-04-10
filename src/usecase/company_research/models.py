from __future__ import annotations

from typing import Any, List, Optional, Union

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.models import Query


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
    user_intent_long:str=Field(
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
        description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
    )
    recommended_queries: Optional[List[Query]] = Field(
        description="Optional list of helpful queries for searching the web to satisfy user intent."
    )


class CriticalAnalysis(BaseModel):
    """result_type for Critic -- a critical analysis of knowledge base."""
    analysis:str= Field(
        description="Critical analysis of gaps, biases, and contradictions in knowledge base."
    )
    new_titles:List[str] = Field(
        description="List of new titles that Critic added to knowledge base."
    )
    recommended_queries: Optional[List[Query]] = Field(
        description="Optional list of search-queries that may further strengthen research."
    )
    

## --  Dependencies for Agents:RunContext --

@dataclass
class ResearchAssistantDeps:
    """Dependencies for the ResearchAssistantAgent."""
    docs: List[Doc] = field(default_factory=list)  # Use default_factory=list
    disambiguation_agent: Optional[Agent] = None # slot for agent help with disambiguation
    report_writer_agent: Optional[Agent] = None # slot for report writer agent
    summarizer_agent: Optional[Agent] = None # slot for summarizing agent (for `add_doc`)
    critic_agent:Optional[Agent] = None # slot for critic


@dataclass
class DisambiguationAgentDeps:
    """Dependencies for the Search Intent Agent."""
    pass


@dataclass
class SummarizerAgentDeps:
    """Dependencies for the Summarizer."""
    pass


@dataclass
class CriticDeps:
    """Dependencies for the Critic Agent."""
    docs: List[Doc] = field(default_factory=list)
    summarizer_agent: Optional[Agent] = None # slot for summarizing agent (for `add_doc`)


@dataclass
class ReportWriterDeps:
    """Dependencies for the ReportWriter Agent."""
    user_intent_long: str = Field(
        description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
    )
    docs: List[Doc] = field(default_factory=list)
        
