from langgraph.graph import StateGraph, END
from app.agents.state import SkillForgeState
from app.agents.planner import planner_agent
from app.agents.resource import resource_agent
from app.agents.quiz import quiz_agent

def build_graph():
    # Create the graph with our state schema
    graph = StateGraph(SkillForgeState)

    # Add nodes — each node is one agent function
    graph.add_node("planner", planner_agent)
    graph.add_node("resource", resource_agent)
    graph.add_node("quiz", quiz_agent)

    # Define edges — the flow between agents
    graph.set_entry_point("planner")       # always start with planner
    graph.add_edge("planner", "resource")  # planner → resource
    graph.add_edge("resource", "quiz")     # resource → quiz
    graph.add_edge("quiz", END)            # quiz → done

    return graph.compile()

# Compile once at module level — reused across requests
skillforge_graph = build_graph()