import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

# -----------------------------
# Configuração LangSmith (precisa já estar no ambiente)
# -----------------------------
# os.environ["LANGCHAIN_API_KEY"] = "sua_chave_langsmith"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "teste-langgraph"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# -----------------------------
# Definição do estado
# -----------------------------
class State(TypedDict):
    question: str
    answer: str

# -----------------------------
# Inicializa LLM via OpenRouter (Gemini)
# -----------------------------
llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",   # modelo do OpenRouter
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------
# Função de nó
# -----------------------------
def answer_question(state: State):
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}

# -----------------------------
# Monta o grafo
# -----------------------------
workflow = StateGraph(State)
workflow.add_node("llm", answer_question)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

# -----------------------------
# Executa com checkpoint em memória
# -----------------------------
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# Teste mínimo
# -----------------------------
if __name__ == "__main__":
    result = app.invoke({"question": "Qual a capital da França?"},
                        config={"configurable": {"thread_id": "teste-1"}})
    print("Resposta:", result["answer"])

