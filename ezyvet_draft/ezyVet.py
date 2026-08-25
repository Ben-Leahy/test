from IPython.display import Image, display
from langchain_openai import ChatOpenAI # TODO change to claude
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
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

# System message
sys_msg = SystemMessage(content="")



# Tool
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
        return END

    tool_called = last_message.tool_calls[0]["name"]
    if tool_called == "sql":
        return "sql"
    elif tool_called == "graphs":            # the tool is named `graphs`...
        return "graph_tool"                  # ...but its node is `graph_tool`
    else: #tool returned was not valid.
        return END

# LLM with bound tool
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY')) # TODO change to claude
llm_with_tools = llm.bind_tools([sql, graphs]) #Note guard is not given here, that is run automatically after sql

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Memory
# Edit this so that it works with postgres instead. 
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# Here is our checkpointer 
# pull file if it doesn't exist and connect to local db
# !mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db
db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("sql", ToolNode([sql]))
builder.add_node("graph_tool", ToolNode([graphs]))  # NOT "graph" — reserved Mermaid keyword
builder.add_node("guard", guard)

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    route_tools,
    # path map: router return value -> node name. Lets LangGraph validate
    # targets AND draw the conditional branches in the graph image.
    {"sql": "sql", "graph_tool": "graph_tool", END: END},
)
builder.add_edge("sql", "guard")
builder.add_edge("guard", "tool_calling_llm")
builder.add_edge("graph_tool", "tool_calling_llm")

# Compile graph
graph = builder.compile(checkpointer = memory)

# To invoke the graph when a user asks a question:
config = {"configurable" : {"thread_id": "1"}} # the thread id should be based on user. Ie, A thread id is only accessible by a single user, 
#and when a user has multiple conversations as there is the ui for in Chat page, then that corresponds to multiple threads.
user_input = ""
input_message = [HumanMessage(content = user_input)]
messages = sys_msg + input_message
graph.invoke(messages, config)

# save image of graph (best-effort — never let drawing crash the app).
try:
    png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Wrote graph.png")
except Exception as e:
    print(f"Could not render graph PNG ({e}); mermaid source below:\n")
    print(graph.get_graph().draw_mermaid())