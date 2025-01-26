from __future__ import annotations
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field

class Query(BaseModel):
    """A text query used for web search or QA datasets."""
    text: str = Field(
        description="A natural language question or keyword terms for a query."
    )

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text


class SearchResult(BaseModel):
    """Represents a single search result from a search engine."""
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

