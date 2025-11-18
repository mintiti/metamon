#!/usr/bin/env python
"""
Phase 1: Verify Critic Access
Test that we can access the critic network from pretrained AMAGO agents.
"""
import sys
import torch
import numpy as np

print("=" * 70)
print("PHASE 1: VERIFYING CRITIC ACCESS")
print("=" * 70)

# Test if we can import necessary modules
try:
    print("\n[1/6] Importing metamon...")
    from metamon.rl.pretrained import SmallRL, get_pretrained_model
    from metamon.interface import UniversalState, DefaultObservationSpace
    print("✓ Metamon imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nNote: You may need to install dependencies:")
    print("  pip install -e .")
    sys.exit(1)

try:
    print("\n[2/6] Importing AMAGO...")
    import amago
    print(f"✓ AMAGO version: {amago.__version__}")
except ImportError as e:
    print(f"✗ AMAGO not installed: {e}")
    print("\nNote: Install AMAGO from: https://ut-austin-rpl.github.io/amago/")
    sys.exit(1)

# Try to initialize a small model (no download yet, just class)
try:
    print("\n[3/6] Creating SmallRL model instance...")
    model = SmallRL()
    print(f"✓ Model created: {type(model).__name__}")
    print(f"  - Observation space: {type(model.observation_space).__name__}")
    print(f"  - Action space: {type(model.action_space).__name__}")
    print(f"  - Tokenizer: {model.tokenizer.vocab_size} vocab size")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Try to initialize the agent (this will download if needed)
try:
    print("\n[4/6] Initializing agent (may download checkpoint ~200MB)...")
    print("  Note: This uses checkpoint 0 (random init) to avoid large download")
    agent = model.initialize_agent(checkpoint=0, log=False)
    print(f"✓ Agent initialized: {type(agent).__name__}")
except Exception as e:
    print(f"✗ Failed to initialize agent: {e}")
    print("\nThis might be due to:")
    print("  - Missing METAMON_CACHE_DIR environment variable")
    print("  - Network issues downloading checkpoint")
    print("  - Missing dependencies")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check agent structure
try:
    print("\n[5/6] Inspecting agent structure...")

    # Check for actor
    has_actor = hasattr(agent, 'actor')
    print(f"  {'✓' if has_actor else '✗'} agent.actor: {has_actor}")
    if has_actor:
        print(f"    Type: {type(agent.actor).__name__}")
        print(f"    Module: {type(agent.actor).__module__}")

    # Check for critic
    has_critic = hasattr(agent, 'critic')
    print(f"  {'✓' if has_critic else '✗'} agent.critic: {has_critic}")
    if has_critic:
        print(f"    Type: {type(agent.critic).__name__}")
        print(f"    Module: {type(agent.critic).__module__}")

        # Check if it's NCritics (ensemble)
        critic_type_name = type(agent.critic).__name__
        if 'NCritics' in critic_type_name or 'Critic' in critic_type_name:
            print(f"    ✓ Critic found!")

            # Try to see if we can access ensemble components
            if hasattr(agent.critic, 'critics'):
                print(f"    Number of critics in ensemble: {len(agent.critic.critics)}")

    # Check for other relevant attributes
    has_policy = hasattr(agent, 'policy')
    print(f"  {'✓' if has_policy else '✗'} agent.policy: {has_policy}")

    has_traj_encoder = hasattr(agent, 'traj_encoder')
    print(f"  {'✓' if has_traj_encoder else '✗'} agent.traj_encoder: {has_traj_encoder}")

    has_tstep_encoder = hasattr(agent, 'tstep_encoder')
    print(f"  {'✓' if has_tstep_encoder else '✗'} agent.tstep_encoder: {has_tstep_encoder}")

    # List all non-private attributes
    print("\n  All agent attributes (non-private):")
    attrs = [a for a in dir(agent) if not a.startswith('_')]
    for attr in sorted(attrs)[:20]:  # Show first 20
        print(f"    - {attr}")
    if len(attrs) > 20:
        print(f"    ... and {len(attrs) - 20} more")

except Exception as e:
    print(f"✗ Error inspecting agent: {e}")
    import traceback
    traceback.print_exc()

# Try to check critic forward signature
try:
    print("\n[6/6] Checking critic forward signature...")

    if has_critic:
        import inspect

        # Get the forward method
        if hasattr(agent.critic, 'forward'):
            forward_sig = inspect.signature(agent.critic.forward)
            print(f"  Critic forward signature: {forward_sig}")

            # Get parameters
            params = forward_sig.parameters
            print(f"  Parameters:")
            for param_name, param in params.items():
                print(f"    - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")

        # Check for other methods
        critic_methods = [m for m in dir(agent.critic) if not m.startswith('_') and callable(getattr(agent.critic, m))]
        print(f"\n  Critic methods (non-private):")
        for method in sorted(critic_methods)[:15]:
            print(f"    - {method}")

        print("\n✓ Critic structure analyzed successfully")
    else:
        print("  ⚠ No critic found - cannot analyze")

except Exception as e:
    print(f"✗ Error analyzing critic: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)

if has_critic:
    print("✓ SUCCESS: Critic network is accessible!")
    print("\nNext Steps:")
    print("  → Phase 2: Understand how to call critic.forward()")
    print("  → Phase 3: Implement top-k value reranking")
else:
    print("✗ FAILED: Could not find critic network")
    print("\nPossible issues:")
    print("  - Agent might use different attribute name")
    print("  - Critic might be embedded in policy")
    print("  - Need to look deeper into AMAGO structure")

print("=" * 70)
