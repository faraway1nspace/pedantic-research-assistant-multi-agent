from dotenv import load_dotenv

# load llm credentials
load_dotenv()

import asyncio
import logging

from src.usecase.company_research.agents import (
    search_intent_agent,
    summarizer_agent,
    critic_agent,
    report_writer_agent,
    research_assistant_agent
)
from src.usecase.company_research.models import ResearchAssistantDeps

# attach muttiple agents to main agent (research assistant)
deps = ResearchAssistantDeps(
    docs=[],
    disambiguation_agent=search_intent_agent,
    report_writer_agent=report_writer_agent,
    summarizer_agent=summarizer_agent,
    critic_agent=critic_agent
)

# info logging
logging.getLogger().setLevel(logging.INFO)


## --- Example usage---
out = research_assistant_agent.run_sync(
    "I'd like a small report for investors about Celestica (TSX listed), such as: what are its various product lines, does it grow organically or through acquisitions, who are its competitors, how concentrated is it's revenue into single-source or single-client contracts like governments.",deps=deps
)
print(result0.data)
# 

    
    
out = research_assistant_agent.run_sync(
    "I'd like a small report for investors about Thomson Reuters (TSX listed), such as: who are its competitors, what are analysts saying about it's growth potential.",deps=deps
)
print(out.data)
## questions="Could you specify what aspects of Thomson Reuters' competitors' growth potential you are most interested in? For example, are you looking for financial growth metrics, market share analysis, or strategic initiatives? Additionally, do you have specific competitors in mind that you would like to focus on?"


out = research_assistant_agent.run_sync(
    "I'm specifically interested in Thomson Reuter's potential for organic revenue growth, as compared to it's main competitors in the legal and tax services domains. Please first figure out its main legal and tax competitors before conducting research.",
    deps=deps,
    message_history=out.new_messages())
)
print(out.data)


# Overview of Celestica Inc.
#  Celestica Inc. is a prominent electronics manufacturing services (EMS) provider that offers a wide range of services including printed circuit assembly, system assembly, and post-manufacturing support. Catering to sectors like computer and communications, Celestica serves major original equipment manufacturers (OEMs) with both low-volume, high-complexity custom products as well as high-volume commodity offerings. The company's diverse product lines and strategic positioning in the EMS sector play a crucial role in addressing the varying needs of its customers.  

# In terms of financial performance, Celestica has demonstrated substantial growth. In 2023, the company reported annual revenue of $7.961 billion, marking a 9.81% increase from 2022. Furthermore, the revenue for the twelve months ending September 30, 2024, reached $9.241 billion, reflecting a 17.52% increase year-over-year. This consistent upward trend in revenue contrasts sharply with many of its competitors, who faced an average contraction of approximately 8.1% in revenue during the same period.  

# Celestica's profitability also stands out; with a net margin of 3.07%, it achieved higher profitability than its competitors. This robust financial performance is supported by a promising strategy focused on product improvement and innovation, particularly concerning the demand for services related to artificial intelligence, which is anticipated to drive the company's growth further in the coming years.  

# In a competitive landscape, Celestica has established a strong position. The company not only outperformed many of its competitors in revenue growth but also positions itself as a partner of choice for major clients in the technology sector. While specific data on client concentration is not detailed in available documents, understanding the diverse nature of Celestica’s offerings suggests that the company spans multiple segments of the computer and technology industry, likely reducing dependency on any single client.  

# Overall, Celestica's effective growth strategies, emphasis on quality service provision, and significant revenue progress underline its crucial role within the EMS industry. Future success will depend on its continued ability to innovate, adapt, and expand its market share amidst evolving industry dynamics.

# Notes:
# [1]: Celestica revenue for the quarter ending September 30, 2024. [link](https://www.macrotrends.net/stocks/charts/CLS/celestica/revenue)
# [2]: Revenue increase and profitability in comparison to competitors. [link](https://csimarket.com/stocks/compet_glance.php?code=CLS)
# [3]: CEO statement on 2023 financial results. [link](https://corporate.celestica.com/news-releases/news-release-details/celestica-announces-fourth-quarter-2023-financial-results)



