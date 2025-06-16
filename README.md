# ai-engine
This is a FastAPI and CLI AI Engine that uses machine learning and LLMs to analyze workloads and generate recommendations with explanations for efficient workload migration between private and public clouds.

## Installation

```bash
make setup
```

## Usage

### API

Run API
```bash
make api
```

Turn on recommendations
```bash
curl -X POST http://0.0.0.0:8083/start
```

Turn off recommendations
```bash
curl -X POST http://0.0.0.0:8083/stop
```

### CLI

Run CLI Mode
```bash
make cli
```

### Docker

Build API
```bash
make build-docker-api
```

Build CLI
```bash
make build-docker-cli
```

Run API
```bash
make run-docker-api
```

Run CLI
```bash
make run-docker-cli
```

