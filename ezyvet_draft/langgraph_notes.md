Okay I need to set up an LLM account, but they don't have free tiers anymore so I need to get a work one. I should probably go through the proper approval process then. 

scan ahead so I know what I need in terms of URI and things.

I wonder if Lachie uses langsmith. 

# Simple graph
Ie router. (module 1)

# Memory
Level 1: 
    References: Module 1, Lesson 7 Agent with memory. agent-memory.ipynb
ie agent-memory (module 1) 
    State
Then we have defining a state as a memory - using pydantic (state schema module 2)
Then we can actually change the state throughout. Note that the states need to be a subset of the overall state for this to happen. Because it's how information is transferred. The key is that with the graph, because you can control input and output as subsets of the overall state, you can still restrict an individual node's view. 

# Working with state
Then we have reducers. Ie how to deal with parallel operating. 
Also how do we deal with an expanding state? Consider the state example of just messages. see state-reducers
How do we put this into practice? see trim-filter-messages. 
What about a more complicated summarisation technique?
    Prompt, get ai to make summary, then delete everything but the summary and the latest message. 

# External memory
    Config plus in the compile() function (replacing memory)

# Parallelisation
Important to reduce overhead, since llm calls can be run in parallel. You can control that multiple parallel branches all need to be finished before proceeding. 
We need a reducer to handle how our state can be updated simultaneously. 

# Sub graphs
They can communicate through keys in the state - where the keys are the same, they are accessible in both and refer to the same variable. Where they differ, they are only available where they are defined.


------------------------------ missing a whole bunch here
Long Term Memory:
Semantic -> Facts -> Ie facts about a user. 
    We can store this as a list or json
Episodic -> Memories -> Ie past agent actions given an example 
    (grounding an agent with sample actions is the few shot method. )
Procedural -> Instructions -> Agent's system prompt. 
    LLMs are actually quite good at writing system prompt, iterating with human feedback.

trust_call can be used in place of with_structured_output and it iterates after testing the output to correctly fit the given json structure. It also lends itself to doing json_patches -> not having to rewrite the entire json when updates are made to context. 


------------------------------------------------------------------------
Module 6 - Deployment
------------------------------------------------------------------------
Deployment:
Langgraph library: 
    open source. 
    pip install langgraph
LangGraph CLI: 
    a way to interact with applications built with the langgraph library through the terminal/ command line. 
    For both development and deployment
    langgraph dev
SDKs (python javascript):
    Frontend -> sdk -> langgraph deployment
    Use the sdk if you want to avoid directly using the http requests (it is somewhere abstracted), 
        but you want the langgraph deployment to be independent to the rest of the application. 
    This might be useful if we were deploying multiple apps to the same langgraph, especially if we weren't sure which languages we wanted to connect to it with
    But realisitcally we might just want to customise the graph with different tools and things anyways, so having the langgraph library directly in our backend isn't a problem
    Even if we don't edit the graph, there's still no advantage to it being deployed separately to the rest of the backend. 
http requests:
    unless you have a language other than python or javascript (the support languages through the sdk), I don't know why you would use this. 
    the sdk is the same thing with helper functions and wrappers. 

LangSmith deployment:
    This gives you access to features for double messaging llms, running scheduled jobs etc. 
    This is paid, and the payment tier depends on self managed self hosted infra, mixed, langsmith managed & hosted infra. 
    There is a free tier: Self-Hosted Lite is free, 1 million node uses per year. You just need the LangGraph CLI to interact with it. 

Applications:
Multiple deployments:
    Each can be used on the same langgraph graph, but with different databases and front ends. 
Remote

------------------------------------------------------------------------
Other notes:
------------------------------------------------------------------------

thread id - we use this for a chat. 
Then we can associate multiple threads with a user id. 
save these to postgres. 
We only need to limit the number of chats per thread, because that is what is associated with token usage. 
I can pass userId in through config

`user_input = "what ToDo should I focus on first."
async for chunk in client.runs.strean(thread["thread_id"].
graph_nane, input=("nessages" :
conf1g=cont1g.l
stream_node="nessages-tuple"):
[HumarMessage (content-user_input) |}.
if chunk-event « "Bessages":
print("', join(data_iten('content'] for data_iten in chunk.data if 'content' in data_item), end="", flush»True)`

long term memory:
store functionality?
Do I want to use the sdk docs for this?