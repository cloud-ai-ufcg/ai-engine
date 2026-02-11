# AI Engine

The AI Engine is the intelligence layer of the cloud simulator that analyzes workload metrics and generates migration recommendations using Large Language Models (LLMs). It processes real-time cluster and workload data from the Monitor and produces optimized placement decisions based on performance, cost, and resource availability.

## Overview

The AI Engine serves as the decision-making core of the simulator, leveraging LLMs to analyze complex multi-cluster scenarios and recommend optimal workload placements. It can operate with different AI models and agent architectures, providing flexibility in decision-making strategies from simple single-agent analysis to sophisticated multi-agent consensus systems.

### Operating Modes

The AI Engine supports two primary agent architectures:

**1. Single Agent Mode**
- Uses a single LLM to analyze all workloads and make migration decisions
- Simpler architecture with faster decision cycles
- Considers performance, cost, and resource constraints holistically
- Suitable for straightforward workload placement scenarios
- Configurable prompts for different optimization strategies

**2. Multi-Agent Mode**
- Employs multiple specialized agents with distinct perspectives:
  - **Performance Agent**: Focuses on resource utilization and cluster load
  - **Cost Agent**: Optimizes for infrastructure costs and cloud pricing
  - **Consolidator Agent**: Synthesizes recommendations from other agents
- Uses LangGraph for orchestrated multi-agent workflows
- Provides more nuanced decision-making with diverse viewpoints
- Suitable for complex scenarios requiring balanced trade-offs

## Prerequisites

- Python 3.12 or higher
- OpenRouter API key (or API keys for supported LLM providers: Google Gemini, OpenAI, Anthropic, NVIDIA)
- MongoDB (for API mode with recommendation history)
- Docker and Docker Compose (for containerized deployment)

## How to Run

> **Note:** For detailed setup and execution instructions, including infrastructure setup and complete workflow, please refer to the [main simulator README](../README.md).

When running as part of the simulator, the AI Engine API is accessible at `http://localhost:8083` and automatically fetches metrics from the Monitor and sends recommendations to the Actuator.

