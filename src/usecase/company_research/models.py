from __future__ import annotations

from typing import Any, List, Optional, Union

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.models import Query
from src.usecase.company_research.config import (
    LLM_NAME, N_PARA_MIN_FOR_REPORT, N_PARA_MAX_FOR_REPORT, N_DOCS_MIN_FOR_REPORT
)

class AskClarifyingQuestionOfUser(BaseModel):
    # DocString: Represents the agent's question for user, to solicit  get clarity and dissambiguate task.
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
   # DocString: represents the final result of research to present to the user
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
   # DocString: Warns the Research Assistant that there aren't enough documents to write a report
    text:str = Field(
        description=(
            f"A text warning to the Research Assistant if there are fewer than {N_DOCS_MIN_FOR_REPORT} docs downloaded to the Knowledge Base."
        )
    

class SearchIntentResult(BaseModel):
    # DocString: Represents the agent's answer in which agent has resolved a user's search intent.
    user_intent_short: str = Field(
        description="A succinct summary of the user's search intent."
    )
    user_intent_long: str = Field(
        description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
    )
    recommended_queries: Optional[List[Query]] = Field(
        description="Optional list of queries helpful for searching documents or the web"
    )


## --  Dependencies for Agents:RunContext --

@dataclass
class ResearchAssistantDeps:
    """Dependencies for the ResearchAssistantAgent."""
    docs: List[Doc] = field(default_factory=list)  # Use default_factory=list
    disambiguation_agent: Optional[Agent] = None # slot for helper agent
    report_writer_agent: Optional[Agent] = None # slot for report writer agent
    summarizer_agent: Optional[Agent] = None # slot for report writer agent


@dataclass
class DisambiguationAgentDeps:
    """Dependencies for the Search Intent Agent."""
    pass


@dataclass
class SummarizerAgentDeps:
    """Dependencies for the Search Intent Agent."""
    pass


@dataclass
class ReportWriterDeps:
    """Dependencies for the ReportWriter Agent."""
    user_intent_long: str = Field(
        description="A detailed outline of the user's intent, scope, desired outputs, and relevant entities."
    )
    docs: List[Doc] = field(default_factory=list)
        
