"""Regenerate graph.png from the CURRENT graph structure — no DB, no LLM calls.

Importing ezyVet only builds `builder` (nodes + edges). The Postgres
connection and checkpointer live under ezyVet's __main__ guard, so they are
NOT run here. The checkpointer never appears in the drawing anyway, so
compiling without one produces an identical image.

Run from this directory:  python draw_graph.py

Note: draw_mermaid_png() renders via the mermaid.ink web API, so this step
needs internet — but it needs no Postgres and no valid Anthropic key.
"""
from ezyVet import builder

# No checkpointer needed just to draw the structure.
graph = builder.compile()

png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
with open("graph.png", "wb") as f:
    f.write(png_bytes)
print("Wrote graph.png")
