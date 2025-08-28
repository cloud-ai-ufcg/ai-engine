## Engine - Multi-agent module

The **AI Engine** has been extended to suport a **multi-agent** architecture using LangGraph (https://www.langchain.com/langgraph).

### 🔹 Motivation

The decision was motivated by the following topics:

- **Specialization**: Increases clarity and madurarity by using specialized agents for each feature to be analized.
- **Flexibility**: New gents can be added easly, keeping the flow scalabe.
- **Traceability**: Easy debugging and observability whith LangSmith tracig.

### 🔹 Architecture Decisions 

- Use of **LangGraph** for flow modeling, as it facilitates orchestration between multiple agents.  
- Integration with **configurable LLMs** via `config.yaml`.  

### 🔹 Architecture 

The multi-agent graph is composed of:

1. **Checker agents** → evaluates some workload criteria such as pending pods.    
2. **Decision agent** → receives votes from experts and decides whether the workload should be **0 - kept** or **1 - migrated**.  
3. **Explainer Agent** → generates a consolidated explanation of the decision, both overall and by workload.

### 🔹 Execution Flow

1. The AI ​​Engine processes workload data (in DataFrame format).  
2. Based on the configuration, it checks whether to use multi-agent mode.  
3. If enabled, the multi-agent graph is invoked.  
4. Model configurations are dynamically loaded.  
5. Each expert agent analyzes the workloads and issues a vote.  
6. The decision-maker agent consolidates the votes and generates the final decision.
7. The explainer agent produces a natural language explanation for each decision.

