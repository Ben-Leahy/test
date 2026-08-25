from IPython.display import Image, display
# from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, ToolMessage
import os

load_dotenv()  # Loads variables from .env

# TODO
"""
Now we want to work with chat history message and stuff. but tbh maybe that's a later change?
 -- We want persistent state
 -- We want external persistence state
 -- We will add a node before END which if message history is > 20, we will summarise the oldest 15, keep that as summary, add 5, add current.

If we get to end then can we just output that to whatever is currently done?
Later: if desired, we can limit the number of calls to a specific tool through state.
"""
# State class to store messages and summary
class State(MessagesState):
    summary: str

# System message
system_prompt = SystemMessage(content="")

config = {"configurable" : {"thread_id": "1"}} # the thread id should be based on user. Ie, A thread id is only accessible by a single user,
#and when a user has multiple conversations as there is the ui for in Chat page, then that corresponds to multiple threads.

# Ask
def ask(user_input, graph, config):
    input_message = [HumanMessage(content = user_input)]
    response = graph.invoke({"messages": input_message}, config)
    return response

def summarize_conversation(state: State):

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

# Tools
def sql(a: int) -> int:
    """Runs sql

    Args:
        a: sql
    """
    return a

def guard(a: int) -> int:
    """Runs guard

    Args:
        a: guard
    """
    return a

def graphs(a: int) -> int:
    """Runs graph

    Args:
        a: graph
    """
    return a

def route_tools(state):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:          # AI produced a plain answer
        return "summarize_conversation"

    tool_called = last_message.tool_calls[0]["name"]
    if tool_called == "sql":
        return "sql"
    elif tool_called == "graphs":            # the tool is named `graphs`...
        return "graph_tool"                  # ...but its node is `graph_tool`
    else: #tool returned was not valid.
        return "invalid_tool_feedback"

# LLM with bound tool
llm = ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv('ANTHROPIC_API_KEY'))
llm_with_tools = llm.bind_tools([sql, graphs]) #Note guard is not given here, that is run automatically after sql

# Node
def tool_calling_llm(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = [SystemMessage(content = f"Summary of conversation earlier: {summary}")]
    else:
        summary_message = []

    messages = [system_prompt] + summary_message + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

# Node - feed a tool_result back for an invalid tool call, then loop to the llm
def invalid_tool_feedback(state: State):
    tool_call = state["messages"][-1].tool_calls[0]
    feedback = ToolMessage(
        content = f"Invalid tool call: '{tool_call['name']}' is not an available tool. Available tools: sql, graphs.",
        tool_call_id = tool_call["id"],
    )
    return {"messages": [feedback]}


# Memory - checkpointer (Postgres).
# The connection is opened only when this file is run directly (see the
# __main__ guard at the bottom), so importing this module — e.g. from
# draw_graph.py to render the graph — never opens a DB connection.
db_uri = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

# Memory - store
# This can be used if we want to save things like user preferences of accumulated knowledge.
# This is for memory lives for a user, independently of the thread id.

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("sql", ToolNode([sql]))
builder.add_node("graph_tool", ToolNode([graphs]))  # NOT "graph" — reserved Mermaid keyword
builder.add_node("guard", guard)
builder.add_node("invalid_tool_feedback", invalid_tool_feedback)
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    route_tools,
    # path map: router return value -> node name. Lets LangGraph validate
    # targets AND draw the conditional branches in the graph image.
    {"sql": "sql", "graph_tool": "graph_tool",
     "invalid_tool_feedback": "invalid_tool_feedback",
     "summarize_conversation": "summarize_conversation"},
)
builder.add_edge("sql", "guard")
builder.add_edge("guard", "tool_calling_llm")
builder.add_edge("graph_tool", "tool_calling_llm")
builder.add_edge("invalid_tool_feedback", "tool_calling_llm")
builder.add_edge("summarize_conversation", END)

# ---------------------------------------------------------------------------
# Everything below runs ONLY when executing this file directly
# (`python ezyVet.py`). It opens the Postgres connection and compiles the
# graph with the checkpointer. Importing this module (e.g. from draw_graph.py)
# skips all of it, so no DB connection is needed just to draw the graph.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langgraph.checkpoint.postgres import PostgresSaver  # separate install from base langgraph

    # connect to the postgres db + create checkpoint tables
    memory = PostgresSaver.from_conn_string(db_uri).__enter__()
    memory.setup()

    # Compile graph with the persistent checkpointer
    graph = builder.compile(checkpointer=memory)

    # Example usage: drive one turn of the graph with a user's message.
    # response = ask("what should I focus on first?", graph, config)

    # save image of graph (best-effort — never let drawing crash the app).
    try:
        png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Wrote graph.png")
    except Exception as e:
        print(f"Could not render graph PNG ({e}); mermaid source below:\n")
        print(graph.get_graph().draw_mermaid())
