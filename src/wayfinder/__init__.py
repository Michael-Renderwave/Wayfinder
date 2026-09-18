"""Wayfinder — a travel research agent with hard retrieval (RAG), ReAct planning,
episodic/semantic memory, and live observability of every search step.

Implementation of the capstone design:
  CP 1.1  travel booking + research assistant, SearchFlight()/BookTicket(), follow-up questions
  CP 2.1  ReAct loop, episodic + semantic memory, tool calling, top-K retrieval
  CP 3.1  recursive character chunking, vector store, similarity ranking, traceable provenance
  CP 4.1  ToT-style candidate tree with beam pruning, rubric scoring
          (source reliability PRIMARY->SECONDARY->TERTIARY, recency, relevance)
"""

__version__ = "1.0.0"
