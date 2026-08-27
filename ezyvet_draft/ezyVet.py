import os

from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
load_dotenv()

# Hard ceiling on LLM calls in a single turn. Once reached, the graph stops
# looping and returns whatever partial text it has (or a generic message).
MAX_LLM_CALLS = 5

# System message
SYSTEM_PROMPT = SystemMessage(content="")

# the thread id should be based on user. Ie, A thread id is only accessible by a
# single user, and when a user has multiple conversations as there is the ui for
# in Chat page, then that corresponds to multiple threads.
CONFIG = {"configurable": {"thread_id": "1"}}

# Postgres connection string for the checkpointer. 
# This should be adapted to a database within the same environment for the rest of the application
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

# Text returned when we hit MAX_LLM_CALLS and have no partial answer to show.
FALLBACK_MESSAGE = (
    "Sorry — I couldn't complete that within the allowed number of steps. "
    "Please try rephrasing your request or breaking it into smaller parts."
)

# =============================================================================
# STATE
# =============================================================================
# State class to store messages and summary
class State(MessagesState):
    summary: str
    llm_calls: int  # per-turn counter; reset to 0 on every `ask` (see below)

# =============================================================================
# TOOLS
# =============================================================================
# Map the current sql generation logic into here
def sql(a: int) -> int:
    """Runs sql

    Args:
        a: sql
    """
    return a

# Pick a graph library that can do histograms, bar charts and pie charts to help responses be visually effective
def graphs(a: int) -> int:
    """Runs graph

    Args:
        a: graph
    """
    return a

# Map the existing guard logic into this function.
def guard(a: int) -> int:
    """Runs guard

    Args:
        a: guard
    """
    return a

# =============================================================================
# CONNECTIONS (LLM & DATABASE)
# =============================================================================
# Map the existing connection logic into the below line. Is the anthropic api key in .env?
llm = ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
llm_with_tools = llm.bind_tools([sql, graphs])

# Memory - store
# This can be used if we want to save things like user preferences of accumulated
# knowledge. This is for memory that lives for a user, independently of the thread id.
# This is not currently implemented, and has no current implementation plans.

# =============================================================================
# NODES
# =============================================================================
# Node
def tool_calling_llm(state: State) -> dict[str, Any]:
    summary = state.get("summary", "")
    if summary:
        summary_message = [SystemMessage(content=f"Summary of conversation earlier: {summary}")]
    else:
        summary_message = []

    messages = [SYSTEM_PROMPT] + summary_message + state["messages"]
    return {
        "messages": [llm_with_tools.invoke(messages)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def summarize_conversation(state: State) -> dict[str, Any]:

    # Only reduce the history once it has grown large enough
    if len(state["messages"]) <= 15:
        return {}

    # First get the summary if it exists
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        # If no summary exists, just create a new one
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 5 most recent messages and add our summary to the state
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-5]]
    return {"summary": response.content, "messages": delete_messages}

# Node - feed a tool_result back for an invalid tool call, then loop to the llm
def invalid_tool_feedback(state: State) -> dict[str, Any]:
    tool_call = state["messages"][-1].tool_calls[0]
    feedback = ToolMessage(
        content=f"Invalid tool call: '{tool_call['name']}' is not an available tool. Available tools: sql, graphs.",
        tool_call_id=tool_call["id"],
    )
    return {"messages": [feedback]}

# Node - reached MAX_LLM_CALLS. Return partial text (or a generic message) and
# stop. The last AI message still has unanswered tool_calls, so we must feed a
# ToolMessage back for each one — otherwise the dangling tool_use would make the
# message history invalid the next time it's sent to Anthropic.
def max_calls_fallback(state: State) -> dict[str, Any]:
    last = state["messages"][-1]

    # Any text the model produced alongside its tool call. Anthropic responses
    # can be a plain string or a list of content blocks, so handle both.
    content = last.content
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    partial = (content or "").strip()

    # Satisfy every pending tool call so the history stays valid next turn.
    tool_msgs = [
        ToolMessage(
            content="Skipped: reached the maximum number of steps.",
            tool_call_id=tc["id"],
        )
        for tc in (last.tool_calls or [])
    ]

    return {"messages": tool_msgs + [AIMessage(content=partial or FALLBACK_MESSAGE)]}

# =============================================================================
# ROUTING
# =============================================================================
def route_tools(state: State) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:          # AI produced a plain answer
        return "summarize_conversation"

    # The AI wants another tool, but we've spent our LLM-call budget — bail out
    # gracefully instead of looping back into tool_calling_llm forever.
    if state.get("llm_calls", 0) >= MAX_LLM_CALLS:
        return "max_calls_fallback"

    tool_called = last_message.tool_calls[0]["name"]
    if tool_called == "sql":
        return "sql"
    elif tool_called == "graphs":            # the tool is named `graphs`...
        return "graph_tool"                  # ...but its node is `graph_tool`
    else: #tool returned was not valid.
        return "invalid_tool_feedback"

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("sql", ToolNode([sql]))
builder.add_node("graph_tool", ToolNode([graphs]))  # NOT "graph" — reserved Mermaid keyword
builder.add_node("guard", guard)
builder.add_node("invalid_tool_feedback", invalid_tool_feedback)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("max_calls_fallback", max_calls_fallback)

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    route_tools,
    # path map: router return value -> node name. Lets LangGraph validate
    # targets AND draw the conditional branches in the graph image.
    {"sql": "sql", "graph_tool": "graph_tool",
     "invalid_tool_feedback": "invalid_tool_feedback",
     "summarize_conversation": "summarize_conversation",
     "max_calls_fallback": "max_calls_fallback"},
)
builder.add_edge("sql", "guard")
builder.add_edge("guard", "tool_calling_llm")
builder.add_edge("graph_tool", "tool_calling_llm")
builder.add_edge("invalid_tool_feedback", "tool_calling_llm")
builder.add_edge("summarize_conversation", END)
# Straight to END (not via summarize) — we're out of budget, so don't spend
# another LLM call summarizing.
builder.add_edge("max_calls_fallback", END)

# =============================================================================
# PUBLIC API
# =============================================================================
def ask(user_input: str, graph: CompiledStateGraph, config: RunnableConfig) -> dict[str, Any]:
    input_message = [HumanMessage(content=user_input)]
    # llm_calls persists in the checkpointer, so reset it to 0 for each new
    # turn — otherwise turn 2 would start already at the cap and bail out.
    response = graph.invoke({"messages": input_message, "llm_calls": 0}, config)
    return response

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================
# Everything below runs ONLY when executing this file directly
# (`python ezyVet.py`). It opens the Postgres connection and compiles the
# graph with the checkpointer. Importing this module (e.g. from draw_graph.py)
# skips all of it, so no DB connection is needed just to draw the graph.
if __name__ == "__main__":
    from langgraph.checkpoint.postgres import PostgresSaver  # separate install from base langgraph

    # connect to the postgres db + create checkpoint tables
    memory = PostgresSaver.from_conn_string(DB_URI).__enter__()
    memory.setup()

    # Compile graph with the persistent checkpointer
    graph = builder.compile(checkpointer=memory)

    # Example usage: drive one turn of the graph with a user's message.
    # response = ask("what should I focus on first?", graph, CONFIG)

    # save image of graph (best-effort — never let drawing crash the app).
    try:
        png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Wrote graph.png")
    except Exception as e:
        print(f"Could not render graph PNG ({e}); mermaid source below:\n")
        print(graph.get_graph().draw_mermaid())
