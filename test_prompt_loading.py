#!/usr/bin/env python3
"""
Simple test script to verify prompt loading from files.
This tests the build_system_prompt_from_config() function with mode-based configuration.
"""

from engine.ai_config import build_system_prompt_from_config, get_agent_mode, get_agent_config
from engine.util import load_config

def test_prompt_loading():
    """Test that prompts are correctly loaded from files for both modes."""
    print("Testing mode-based prompt loading system...\n")
    
    # Load config
    config = load_config()
    current_mode = get_agent_mode()
    
    print(f"Current agent mode: {current_mode}")
    print(f"{'='*60}\n")
    
    # Test single_agent mode
    print("SINGLE AGENT MODE:")
    print("-" * 60)
    single_config = get_agent_config("single_agent")
    single_prompt = build_system_prompt_from_config("single_agent")
    selected = single_config.get("selected_prompt", "N/A")
    print(f"Selected prompt: {selected}")
    print(f"\nPrompt content:")
    print(single_prompt)
    print(f"\n{'='*60}\n")
    
    # Test multi_agent mode
    print("MULTI-AGENT MODE:")
    print("-" * 60)
    multi_config = get_agent_config("multi_agent")
    multi_prompt = build_system_prompt_from_config("multi_agent")
    selected = multi_config.get("selected_prompt", "N/A")
    print(f"Selected prompt: {selected}")
    
    # Show multi-agent specific config
    agents_config = config.get("ai", {}).get("multi_agent", {}).get("agents", {})
    if agents_config:
        print("\nAgent configuration:")
        for agent_name, agent_config in agents_config.items():
            enabled = agent_config.get("enabled", False)
            weight = agent_config.get("weight", 1.0)
            status = "✓ enabled" if enabled else "✗ disabled"
            print(f"  - {agent_name}: {status} (weight: {weight})")
    
    print(f"\nPrompt content:")
    print(multi_prompt)
    print(f"\n{'='*60}\n")
    
    # Verify rules are embedded in prompts
    print("VERIFICATION:")
    print("-" * 60)
    rules_keywords = ["Rules:", "Cluster load", "Pending workloads", "recommendations"]
    
    for mode_name, prompt in [("single_agent", single_prompt), ("multi_agent", multi_prompt)]:
        print(f"\n{mode_name.upper()} prompt contains:")
        for keyword in rules_keywords:
            if keyword in prompt:
                print(f"  ✓ {keyword}")
            else:
                print(f"  ✗ {keyword} (missing)")
    
    print("\n✓ Prompt loading test completed!")
    print("\nConfiguration guide:")
    print("  - Change mode: Set 'ai.mode' to 'single_agent' or 'multi_agent'")
    print("  - Change single_agent prompt: Set 'ai.single_agent.selected_prompt'")
    print("  - Change multi_agent prompt: Set 'ai.multi_agent.selected_prompt'")
    

if __name__ == "__main__":
    test_prompt_loading()
