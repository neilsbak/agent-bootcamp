"""KYC Person Search Agent.

Usage:
    uv run --env-file .env python -m src.examples.kyc_person_search "John Smith, CEO of Acme Corp"
"""

import asyncio
import operator
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from src.utils.tools.gemini_grounding import (
    GeminiGroundingWithGoogleSearch,
    ModelSettings,
)


load_dotenv(verbose=True)


KYC_SEARCH_INSTRUCTIONS = """
You are a KYC (Know Your Customer) research agent that gathers publicly available
information for compliance review. Your role is to FIND and REPORT information -
you do NOT make determinations about fraud or risk.

When given a person's name (and optionally their role/company), conduct thorough
searches to gather information in these categories:

1. **Identity Verification**
   - Current role, title, company
   - Location/jurisdiction
   - Professional credentials

2. **Professional Background**
   - Employment history
   - Industry experience
   - Education (if publicly available)

3. **Legal & Regulatory**
   - Any mentions in lawsuits (plaintiff or defendant)
   - Regulatory actions, sanctions, or enforcement
   - SEC filings, FINRA actions (if applicable)
   - Sanctions lists mentions (OFAC, UN, EU)

4. **Adverse Media / Negative News**
   - Fraud allegations or investigations
   - Criminal proceedings
   - Controversies or scandals
   - Whistleblower complaints

5. **Political Exposure (PEP)**
   - Government positions (current or former)
   - Political party affiliations
   - Relationships to government officials

6. **Business Associations**
   - Companies where they are director/officer
   - Business ownership
   - Bankruptcy filings
   - Shell company associations

**Important Guidelines:**
- Report ONLY what you find - do not speculate or infer
- Clearly distinguish between confirmed facts and allegations
- Include ALL sources with URLs
- Note if searches return no results (absence of findings is also useful)
- Flag any limitations (e.g., common name making results ambiguous)
- Be factual and neutral in tone
"""

SYSTEM_MESSAGE = """You are a KYC (Know Your Customer) research assistant. Your role is to gather
publicly available information on individuals for compliance review.

When asked to research a person:
1. Use the run_person_search tool with their name and any identifying details (company, title, location)
2. Present the findings clearly, organized by category (identity, professional background, legal/regulatory, adverse media, political exposure, business associations)
3. Always include the sources/citations from the search results
4. Note any limitations (e.g., common name, limited public information)

You gather and report information - you do NOT make risk determinations or recommendations.
That is for compliance analysts to decide.
"""

# Initialize the model (using OpenAI-compatible endpoint configured in .env)
model = init_chat_model("openai:gemini-2.5-flash", temperature=0)


@tool
async def run_person_search(name: str) -> str:
    """Run a KYC search on the given subject.

    Args:
        name: The person to search for. Include identifying details
            like company, title, or location.
    """
    gemini_search = GeminiGroundingWithGoogleSearch(
        model_settings=ModelSettings(model="gemini-2.5-flash")
    )
    query = f"""
{KYC_SEARCH_INSTRUCTIONS}

Subject to research: {name}
"""
    search_result = await gemini_search.get_web_search_grounded_response(query)
    return search_result.text_with_citations


# Set up tools
tools = [run_person_search]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    """State for the KYC search agent graph."""

    messages: Annotated[list[AnyMessage], operator.add]


async def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or not."""
    messages = [SystemMessage(content=SYSTEM_MESSAGE)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def tool_node(state: MessagesState) -> dict:
    """Perform the tool call."""
    results = []
    last_message: AIMessage = state["messages"][-1]  # type: ignore[assignment]
    for tool_call in last_message.tool_calls:
        tool_func = tools_by_name[tool_call["name"]]
        observation = await tool_func.ainvoke(tool_call["args"])
        results.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": results}


def should_continue(state: MessagesState) -> str:
    """Decide if we should continue the loop or stop."""
    last_message: AIMessage = state["messages"][-1]  # type: ignore[assignment]
    if last_message.tool_calls:
        return "tool_node"
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python -m src.examples.kyc_person_search '<person name>'")
        print("Example: uv run python -m src.examples.kyc_person_search 'John Smith, CEO of Acme Corp'")
        sys.exit(1)

    subject = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"KYC Search: {subject}")
    print(f"{'='*60}\n")

    messages = [HumanMessage(content=f"Research this person for KYC compliance: {subject}")]
    result = await agent.ainvoke({"messages": messages})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
