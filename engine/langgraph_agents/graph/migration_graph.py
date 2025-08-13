import pandas as pd
from typing import Dict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ...langgraph_agents import (
    cpu_checker,
    mem_checker,
    pending_checker,
    decision_agent,
    explainer_agent
)


# -----------------------
# Estado da simulação
# -----------------------
class MigrationState:
    def __init__(self, workloads: pd.DataFrame):
        self.workloads = workloads
        self.cpu_votes: List[int] = []
        self.mem_votes: List[int] = []
        self.pending_votes: List[int] = []
        self.final_decisions: List[int] = []
        self.explanations: Dict = {}


# -----------------------
# Funções de nós
# -----------------------
def cpu_node(state: MigrationState) -> MigrationState:
    state.cpu_votes = cpu_checker(state.workloads)
    return state


def mem_node(state: MigrationState) -> MigrationState:
    state.mem_votes = mem_checker(state.workloads)
    return state


def pending_node(state: MigrationState) -> MigrationState:
    state.pending_votes = pending_checker(state.workloads)
    return state


def decision_node(state: MigrationState) -> MigrationState:
    state.final_decisions = decision_agent(
        state.cpu_votes, state.mem_votes, state.pending_votes
    )
    return state


def explainer_node(state: MigrationState) -> MigrationState:
    state.explanations = explainer_agent(
        state.workloads,
        {"cpu": state.cpu_votes, "mem": state.mem_votes, "pending": state.pending_votes},
        state.final_decisions
    )
    return state


# -----------------------
# Criação do grafo
# -----------------------
def create_migration_graph():
    graph = StateGraph(MigrationState)

    # Adiciona os nós
    graph.add_node("cpu", cpu_node)
    graph.add_node("mem", mem_node)
    graph.add_node("pending", pending_node)
    graph.add_node("decision", decision_node)
    graph.add_node("explainer", explainer_node)

    # Ligações
    graph.set_entry_point("cpu")
    graph.add_edge("cpu", "mem")
    graph.add_edge("mem", "pending")
    graph.add_edge("pending", "decision")
    graph.add_edge("decision", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile(checkpointer=MemorySaver())


# -----------------------
# Função para executar
# -----------------------
def run_migration_pipeline(workloads_df: pd.DataFrame):
    """
    Executa o pipeline de decisão de migração usando LangGraph.
    Retorna um DataFrame com as decisões e as explicações.
    """
    # Cria o estado inicial
    state = MigrationState(workloads_df)

    # Executa o grafo
    app = create_migration_graph()
    final_state: MigrationState = app.invoke(state)

    # Monta resultado final como DataFrame
    result_df = workloads_df.copy()
    result_df["label"] = final_state.final_decisions

    return result_df, final_state.explanations
