from langgraph.graph import StateGraph, END
from ..nodes.langgraph_agents import cpu_checker
from ..nodes.langgraph_agents import mem_checker
from ..nodes.langgraph_agents import pending_checker
from ..nodes.langgraph_agents import decision_agent
from ..nodes.langgraph_agents import explainer_agent
from engine.data_types import WorkflowState

def run_cpu_checker(state: WorkflowState):
    cpu_votes = cpu_checker(state["workloads"])
    return {"cpu_votes": cpu_votes}

def run_mem_checker(state: WorkflowState):
    mem_votes = mem_checker(state["workloads"])
    return {"mem_votes": mem_votes}

def run_pending_checker(state: WorkflowState):
    pending_votes = pending_checker(state["workloads"])
    return {"pending_votes": pending_votes} 

def run_decision_agent(state: WorkflowState):
    decisions = decision_agent(
        state["cpu_votes"], state["mem_votes"], state["pending_votes"]
    )
    return {"final_decisions": decisions} 

def run_explainer(state: WorkflowState):
    votes = {
        "cpu": state["cpu_votes"],
        "mem": state["mem_votes"],
        "pending": state["pending_votes"]
    }
    explanation = explainer_agent(state["workloads"], votes, state["final_decisions"])
    return {"explanations": explanation} 

def build_graph():
    workflow = StateGraph(WorkflowState)

    # Add the nodes
    workflow.add_node("cpu", run_cpu_checker)
    workflow.add_node("mem", run_mem_checker)
    workflow.add_node("pending", run_pending_checker)
    workflow.add_node("decision", run_decision_agent)
    workflow.add_node("explainer", run_explainer)

    # Parallel entry points
    workflow.set_entry_point("split") 

    
    workflow.add_node("split", lambda state: state)

    
    workflow.add_edge("split", "cpu")
    workflow.add_edge("split", "mem")
    workflow.add_edge("split", "pending")


    workflow.add_edge("cpu", "decision")
    workflow.add_edge("mem", "decision")
    workflow.add_edge("pending", "decision")

    workflow.add_edge("decision", "explainer")
    workflow.add_edge("explainer", END)

    return workflow.compile()

if __name__ == "__main__":
    from utils.parser import load_workloads_from_monitor

    # Loads
    graph = build_graph()

    # Gera imagem do grafo com Mermaid
    mermaid_png = graph.get_graph().draw_mermaid_png()

    # Salva a imagem
    with open("migration_graph.png", "wb") as f:
        f.write(mermaid_png)

    print("✅ graph saved as 'migration_graph.png'")