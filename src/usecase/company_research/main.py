from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from src.usecase.company_research.agents import (
    search_intent_agent,
    summarizer_agent,
    report_writer_agent,
    research_assistant_agent
)
from src.usecase.company_research.models import ResearchAssistantDeps

logging.getLogger().setLevel(logging.INFO)

# load llm credentials
load_dotenv()



