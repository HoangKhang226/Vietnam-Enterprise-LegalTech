from functools import lru_cache
from src.agents.graph import build_graph

@lru_cache()
def get_graph():
    return build_graph()
