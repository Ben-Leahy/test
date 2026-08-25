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
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
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

#====================================
# preparation
#====================================
# State class to store messages and summary
class State(MessagesState):
    summary: str
    
# System message
system_prompt = SystemMessage(content="")

config = {"configurable" : {"thread_id": "1"}} # the thread id should be based on user. Ie, A thread id is only accessible by a single user, 
#and when a user has multiple conversations as there is the ui for in Chat page, then that corresponds to multiple threads.

#====================================
# Call graph
#====================================
# Ask
def ask(state: State, graph, config, model):
    user_input = "" # Wire up to what recieves user input
    input_message = [HumanMessage(content = user_input)]
    summary = state.get("summary", "")
    if summary:
        summary_message = [SystemMessage(content = f"Summary of conversation earlier: {summary}")]
    else: 
        summary_message = []

    messages = [system_prompt] + summary_message + state["messages"] + input_message
    return None

#====================================
# Nodes
#====================================
def summarize_conversation(state: State, model):
    
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
    response = model.invoke(messages)
    
    # Delete all but the 4 most recent messages and add our summary to the state 
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
    return {"summary": response.content, "messages": delete_messages}

def tool_calling_llm():
    return {"messages": [llm_with_tools.invoke([system_prompt] + state["messages"])]}

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
        return END

    tool_called = last_message.tool_calls[0]["name"]
    if tool_called == "sql":
        return "sql"
    elif tool_called == "graphs":            # the tool is named `graphs`...
        return "graph_tool"                  # ...but its node is `graph_tool`
    else: #tool returned was not valid.
        return END

# LLM with bound tool
llm = ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv('ANTHROPIC_API_KEY'))
llm_with_tools = llm.bind_tools([sql, graphs]) #Note guard is not given here, that is run automatically after sql


# Memory - checkpointer
# Edit this so that it works with postgres instead. 
from langgraph.checkpoint.postgres import PostgresSaver # this is a separate install to the basic langgraph so needs to be in requirements
# Here is our checkpointer
# connect to the postgres db
db_uri = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
memory = PostgresSaver.from_conn_string(db_uri).__enter__()
memory.setup()

# Memory - store
# This can be used if we want to save things like user preferences of accumulated knowledge. 
# This is for memory lives for a user, independently of the thread id. 

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", ask)
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



# save image of graph (best-effort — never let drawing crash the app).
try:
    png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Wrote graph.png")
except Exception as e:
    print(f"Could not render graph PNG ({e}); mermaid source below:\n")
    print(graph.get_graph().draw_mermaid())