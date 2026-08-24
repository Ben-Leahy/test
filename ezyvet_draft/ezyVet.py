from IPython.display import Image, display
from langchain_openai import ChatOpenAI # TODO change to claude
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env

# TODO
"""
I want to see if I can have sql as a tool call, then it just has to go through the guard next. regardless of whether it passes or fails
it should go back to the agentic node that decides what to do next. 
    So we are going to add a new node which is guard, and add an edge from sql to guard, guard to tool calling node.

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

# LLM with bound tool
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY')) # TODO change to claude
llm_with_tools = llm.bind_tools([sql, graphs]) #Note guard is not given here, that is run automatically after sql

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("sql_tool", ToolNode([sql]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("sql_tool", "tool_calling_llm") 

# Compile graph
graph = builder.compile()

# save image of graph.
png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)