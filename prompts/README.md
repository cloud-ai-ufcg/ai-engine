# Prompt Versioning

This directory contains versioned system prompts for the AI engine. Prompts are stored as separate text files to enable version control, testing, and easy switching between different prompt strategies.

**Important**: Rules are now embedded directly in the prompt files, not in the config file.

## Available Prompts

### Single Agent Prompts

- **`single_agent.txt`** - Base single agent prompt (legacy, not recommended)
- **`single_agent_v1.txt`** - Version 1: Basic workload migration advisor with standard rules
- **`single_agent_v2.txt`** - Version 2: Enhanced with resource optimization focus and additional rules

### Multi-Agent Prompts

- **`multi_agent.txt`** - Prompt for multi-agent system collaboration with coordination rules

## Configuration Structure

The AI engine now uses a **mode-based configuration** with separate settings for single-agent and multi-agent modes:

```yaml
ai:
  mode: single_agent  # 'single_agent' or 'multi_agent'
  
  single_agent:
    enabled: true
    selected_prompt: single_agent_v1  # Prompt for single-agent mode
    provider: openrouter
    generation_config:
      temperature: 0.1
      max_output_tokens: 12000
  
  multi_agent:
    enabled: false
    selected_prompt: multi_agent      # Prompt for multi-agent mode
    provider: openrouter
    coordination_strategy: voting
    generation_config:
      temperature: 0.1
      max_output_tokens: 12000
    agents:
      cpu_checker:
        enabled: true
        weight: 1.0
      memory_checker:
        enabled: true
        weight: 1.0
      pending_checker:
        enabled: true
        weight: 1.5
```

## Usage

### Switching Between Modes

To switch between single-agent and multi-agent modes:

```yaml
ai:
  mode: single_agent  # Change to 'multi_agent' to use multi-agent mode
```

### Changing Prompts

For **single-agent mode**:
```yaml
ai:
  single_agent:
    selected_prompt: single_agent_v2  # Change to any available single-agent prompt
```

For **multi-agent mode**:
```yaml
ai:
  multi_agent:
    selected_prompt: multi_agent  # Change to any available multi-agent prompt
```

## Creating New Prompts

1. **Create the prompt file**: Add a new `.txt` file in this directory
2. **Include rules**: Embed all rules and constraints directly in the prompt text
3. **Format**: Use plain text with clear sections (e.g., "Rules:")
4. **Update config**: Reference your new prompt in the appropriate mode configuration
5. **Test**: Use `python test_prompt_loading.py` to verify loading
6. **Document**: Update this README with the new prompt details

### Example Prompt Structure

```
You are an expert [role description]. [Task description].

Rules:
- Rule 1
- Rule 2
- Rule 3

Additional guidelines:
- Guideline 1
- Guideline 2
```

## Best Practices

- **Embed rules**: All rules must be in the prompt file, not in config
- **Version your prompts**: Use `_v1`, `_v2`, etc. suffixes for iterations
- **Mode-specific prompts**: Single-agent and multi-agent prompts should have different instructions
- **Document changes**: Update this README when adding new prompts
- **Keep it simple**: Store only the prompt text, no markup or metadata
- **Test thoroughly**: Validate new prompts before production use
- **Name descriptively**: Use clear names that indicate the prompt's purpose and version

## Prompt Guidelines

When creating new prompts, consider:

- **Clarity**: Be explicit about the AI's role and responsibilities
- **Context**: Include domain-specific knowledge (e.g., Kubernetes expertise)
- **Rules**: Embed all rules, constraints, and guidelines directly in the prompt
- **Mode-awareness**: Tailor prompts for single-agent vs multi-agent coordination
- **Output format**: Specify expected response structure if needed
- **Versioning**: Increment version numbers for iterations, keep old versions for rollback

## Testing

Run the test script to verify prompt loading:

```bash
python test_prompt_loading.py
```

This will show:
- Current agent mode
- Loaded prompts for both single-agent and multi-agent modes
- Verification that rules are embedded in the prompts
- Configuration guidance
