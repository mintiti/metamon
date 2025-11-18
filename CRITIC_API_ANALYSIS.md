# AMAGO Critic API Analysis

## Based on Code Inspection

### Critic Structure

From `/metamon/rl/configs/models/small_agent.gin`:
```gin
Agent.critic_type = @actor_critic.NCritics
Actor_critic.NCritics.activation = "leaky_relu"
actor_critic.NCritics.n_layers = 2
actor_critic.NCritics.d_hidden = 300
Agent.num_critics = 4
```

**Conclusion:** The critic is `amago.nets.actor_critic.NCritics` - an ensemble of 4 Q-networks.

### AMAGO Agent Forward Pass (Inferred)

AMAGO agents follow this pattern:
1. **Timestep Encoding**: Raw observation → embedded representation
2. **Trajectory Encoding**: Sequence of timesteps → contextual state
3. **Actor**: State → Action logits
4. **Critic**: State → Value estimates (Q-values for each action)

### Critic Input/Output

Based on standard AMAGO patterns, the critic likely has this interface:

```python
# Option A: Critic computes Q(s,a) for all actions at once
q_values = agent.critic(state_embedding)  # Shape: [batch, num_actions]

# Option B: Critic takes state and action separately
q_value = agent.critic(state_embedding, action)  # Shape: [batch, 1]
```

**Most likely:** Option A (compute all Q-values), because:
- More efficient (one forward pass)
- Standard in discrete action spaces
- NCritics returns ensemble values

### NCritics Ensemble

With 4 critics, `agent.critic(state)` returns:
- Shape: `[batch, num_critics=4, num_actions]`
- We can aggregate: `mean()`, `min()`, or `max()`

## Key Discovery: AMAGO's `act()` Method

Looking at AMAGO agent patterns, the `act()` method:
1. Processes observations through encoders
2. Calls actor to get logits
3. Samples action from logits
4. **Does NOT use critic at inference time by default**

This is our opportunity! We can modify to use critic.

## Implementation Strategy

### Approach: Modify agent's action selection

```python
# Instead of:
action = agent.act(obs)  # Just uses actor

# We'll do:
action = value_reranked_act(agent, obs, k=5)  # Uses actor + critic
```

### How to Access State Embeddings

AMAGO agents maintain hidden states for recurrent encoders. We need to:
1. Process observation through `agent.tstep_encoder`
2. Update trajectory with `agent.traj_encoder`
3. Get state embedding from trajectory encoder output
4. Use this state embedding for both actor AND critic

**Key insight:** We need to call the agent's internal encoding pipeline, not just the actor directly.

## Next Steps

1. Create a wrapper that intercepts `agent.act()`
2. Extract state embedding from the encoding pipeline
3. Get top-k actions from actor logits
4. Evaluate with critic
5. Return best action

## Potential Challenges

1. **Hidden State Management**: Need to maintain RNN states correctly
2. **Batch Dimensions**: AMAGO uses [Batch, Length, ...] shapes
3. **Action Masking**: Must apply illegal action mask before selecting

## Solution: Hook into Agent's Internals

We'll create a modified agent class that overrides the action selection:

```python
class ValueRerankingAgent(Agent):
    def __init__(self, base_agent, k=5, aggregation='mean'):
        self.base_agent = base_agent
        self.k = k
        self.aggregation = aggregation

    def act(self, obs):
        # 1. Process obs to get state embedding
        # 2. Get actor logits
        # 3. Top-k from policy
        # 4. Evaluate with critic
        # 5. Return best
```
