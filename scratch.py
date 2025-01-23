from dotenv import load_dotenv

# load llm credentials
load_dotenv()

import asyncio
import logging

from src.usecase.company_research.agents import (
    search_intent_agent,
    summarizer_agent,
    report_writer_agent,
    research_assistant_agent
)
from src.usecase.company_research.models import ResearchAssistantDeps

# attach muttiple agents to main agent (research assistant)
deps = ResearchAssistantDeps(
    docs=[],
    disambiguation_agent=search_intent_agent,
    report_writer_agent=report_writer_agent,
    summarizer_agent=summarizer_agent
)

## Example usage
out = research_assistant_agent.run_sync(
    "I'd like a small report for investors about Celestica (TSX listed), such as: what are its various product lines, does it grow organically or through acquisitions, who are its competitors, how concentrated is it's revenue into single-source or single-client contracts like governments.",deps=deps
)
print(result0.data)
# 

    
    
