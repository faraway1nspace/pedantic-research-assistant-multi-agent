# pedantic-research-assistant-multi-agent

## DEMO: Build Your Own Deep-Research Assistant with the Pydantic-AI Multi-Agent Framework

Google's multi-agent [Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) is making waves and in tech-circles: it's an LLM research assistant that can be given a research task, search the web, manipulate data, critique it's intermeidate steps, collect citations, and generate high-quality reports that would previously take days if not weeks to assemble.

In this repo, we'll walk through a [DIY multi-agent Research Assistant](https://github.com/faraway1nspace/pedantic-research-assistant-multi-agent), using the new [Pydantic-AI](https://ai.pydantic.dev/) framework. You'll learn how to build LLM-powered multi-agent workflows, to perform web searches, download webpages and PDFs, summarize lengthy documents, and synthesize the information into a concise research report -- maybe you'll even get a little productivity boost from this Deep Research alternative!

<blockquote>
  <strong>Demo Use Case:</strong> We'll demonstrate Pydantic-AI to help us with researching publicly-traded company for investment insights, due diligence, competitor analysis, and/or get the dirt before applying for a job. <b>You can modify the use-case by changing the `system` prompts.
</blockquote>

### Before You Get Started

- You'll need an API key from either OpenAI (environment variable name `OPENAI_API_KEY`) or Anthropic. Create a `.env` file in your project's root directory and add your API key there. You can find instructions on how to obtain an OpenAI API key in the OpenAI documentation: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- This project has been tested on Python 3.11.
- Install the [required packages](https://github.com/faraway1nspace/pedantic-research-assistant-multi-agent/blob/main/requirements.txt) using the following command:
```bash
pip install -r requirements.txt
```
- **Warning**: You might need to install the `duckduckgo_search` package separately using the following command to avoid rate-limitations: `pip install --upgrade --quiet duckduckgo-search`

## Pydantic-AI: A Primer

Pydantic-AI is a Python library built upon the popular Pydantic library, known for its data validation, type hinting, and serialization features. Pydantic-AI extends this functionality to the realm of LLMs, allowing you to define AI agents that can interact with Python tools. 

Here are some core concepts in Pydantic-AI:

- **Agents:** AI agents are the heart of Pydantic-AI. They encapsulate an LLM (like GPT-4o-mini) and can be given special personalities via their `system_message` argument. They can access tools, execute code, and generate outputs based on defined data models.
- **Tools:** Tools allow agents to interact with the broader digital world. These can  from simple deterministic functions (like searching DuckDuckGo) to more complex tools that wrap other AI agents, allowing for hierarchical multi-agent flows.
- **Results:** Pydantic-AI leverages Pydantic's strength in data validation by allowing you to define the expected output schema of an agent using Pydantic models. This ensures predictability and structure for agent responses.
    - e.g., by defining a `Footnote` class as part of the final `ResearchReport`, the report writing agent is primed to collect citations and write them in a consistent format according to `Footnote`.
- **Dependencies:** Dependencies are dataclasses that hold shared objects, allowing agents and tools to access them. This facilitates communication and collaboration between agents. For instance, in our research assistant, we use dependencies to create a shared "knowledge base" where web-pages and PDFs are downloaded for report writing agent.
- **Typing:** Well-typed tools make it easier for agents to understand how to execute a function and what info it requires.

Pydantic-AI also makes extensive use of decorators. For example, there is the `@agent.tool` decorator that registers a Python function to an agent, letting it know it is available to use.
    ```python
    @math_agent.tool
    async add(ctx: RunContext[MathDependencies], a:float, b:float)->float:
        """Add two values together."""
        return a*b
    ```
    
Similarly, you can use the `@agent.system_prompt` decorator to dynamically add content to an agent's system prompt, tailoring its behavior to a user's context and situation.

    ```python
    @chat_agent.system_prompt
    async def add_current_date(ctx: RunContext[ChatDependencies]) -> str:
        """Add today's date to the chatbot's system prompt"""
        return f"Today's date and time is: {datetime.now()}"
    ```


## Overview of the Research Assistant: A Multi-Agent Workflow

You might wonder, why use multiple agents instead of a single, all-encompassing agent? Here are some reasons to break down a project into multiple task-focused agents:
- **Specialization:** Different agents can have specialized system prompts and tools, making them more adept at specific tasks. E.g., we can tailor agents to different scenarios via few-shot in-context learning..
- **Modular Design:** It is easy to expand functionality by reusing agent-code. In this demo, we'll show how to expand our research assistant by adding a 'critic' agent.
     
Here's a breakdown of the various agents in this demo:

1. **Research Assistant Agent (Main Agent):** This agent orchestrates the entire workflow. It interacts with the user, delegates tasks to sub-agents, performs web searches, and decides which documents to download. It also has access to a "knowledge base" where it stores the downloaded documents (implemented using dependencies).
2. **Search Intent Agent:** This agent acts as a sanity check on user-queries. It interacts with the user to clarify their research intent and recommend relevant search queries. This agent ensures that subsequent web searches are more focused and effective.
3. **Summarizer Agent (Optional):** This agent is more of a specialized tool than a full-fledged agent. It condenses large documents into more manageable summaries for efficient processing. It helps reduce the cognitive load on the report writer.
4. **Report Writer Agent:** This agent takes the gathered information from the knowledge base and synthesizes it into a well-structured research report. It focuses on extracting key insights, maintaining objectivity, and properly citing sources using footnotes.

## EXAMPLE

Let's research Thomson Reuter's legal and tax business for potential investors..

```python
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
    "I'd like a small report for investors about Thomson Reuters (TSX:TRI), such as: who are its competitors, what are analysts saying about it's growth potential.", deps=deps
)
print(out.data)
## AskClarifyingQuestionOfUser(questions="Could you specify what aspects of Thomson Reuters' competitors' growth potential you are most interested in? For example, are you looking for financial growth metrics, market share analysis, or strategic initiatives? Additionally, do you have specific competitors in mind that you would like to focus on?")

## The first response was type `AskClarifyingQuestionOfUser`, whereby the agent wanted clarification about our research interest in Thomson Reuters.
## We reply with a more refined query.

## To continue a conversation, add the previous chat history via the `message_history` argument. 
out = research_assistant_agent.run_sync(
    "I'm specifically interested in Thomson Reuter's potential for organic revenue growth, as compared to it's main competitors in the legal and tax services domains. Please first figure out its main legal and tax competitors before conducting research.",
    deps=deps,
    message_history=out.new_messages()) # ADD MESSAGE HISTORY to continue chat
)
print(out.data)
# Thomson Reuters Revenue Growth Potential Among Competitors in Legal and Tax Services
#  Thomson Reuters (TRI), a leading provider of legal and tax services, has positioned itself strategically to harness organic revenue growth, particularly through innovations in technology such as generative AI. The company reported an impressive third-quarter revenue of $1.72 billion in 2023, surpassing analyst expectations, and projects a 7% organic revenue increase for the year, up from an earlier estimate of 6.5%[^1]. This growth is significantly influenced by advancements in AI-driven products, like Westlaw AI and CoCounsel, aimed at assisting legal professionals in research and documentation processes.
# In the competitive landscape, Thomson Reuters faces strong competition from key players such as Bloomberg, LexisNexis, and Dow Jones, which provide comprehensive solutions within the legal and tax sectors. LexisNexis, for instance, is recognized for its extensive database in legal research, while Bloomberg adds value through its integrations of diverse legal content and analytics tools. Dow Jones, known for The Wall Street Journal, commands a substantial audience, enhancing its capabilities in legal and financial news distribution[^2]. These competitors are also innovating their service offerings in order to capture market share, thus intensifying the competition in organic revenue growth across the sector.
# Looking ahead, Thomson Reuters is committed to increasing its investment in AI technologies, with a budget exceeding $200 million aimed at expanding its capabilities further in 2024. This targeted investment signals a proactive approach to capturing growth in areas underserved in the legal and tax services market[^3]. Analysts maintain a cautiously optimistic outlook for Thomson Reuters as it leverages these innovations to strengthen its offerings and enhance user loyalty. Comparative analysis indicates that while competitors are also exploring technological innovations and market expansions, Thomson Reuters appears well-positioned relative to its existing user base and market reputation, which could lead to higher organic growth rates in the coming years.
# In conclusion, Thomson Reuters is set for continued organic revenue growth within the legal and tax services market, supported by robust AI investments and a solid product portfolio. As competitors like LexisNexis and Bloomberg enhance their offerings, the marketplace will demand continuous innovation and adaptability. Thus, the environment promises considerable opportunities for all players involved as they strive to meet the evolving needs of clients.
# Notes:
# [1]: Thomson Reuters reported a 7% increase in organic revenue expectations for the year based on third-quarter results. [link](https://finance.yahoo.com/news/thomson-reuters-reports-higher-third-113912340.html)
# [2]: Competitors like Bloomberg, LexisNexis, and Dow Jones are key players in the legal and tax sectors. [link](https://thebigmarketing.com/thomson-reuters-competitors/)
# [3]: Thomson Reuters is increasing its AI investment to over $200 million in 2024 to enhance its service offerings. [link](https://finance.yahoo.com/news/thomson-reuters-reports-higher-third-113912340.html)
``` 