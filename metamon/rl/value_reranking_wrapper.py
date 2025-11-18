"""
Value Reranking Wrapper for AMAGO Agents

Wraps an existing AMAGO agent to use value reranking during action selection.
Compatible with existing evaluation code (evaluate.py).
"""
import torch
import warnings
from typing import Optional, Literal

from metamon.rl.inference_strategies import ValueRerankingStrategy, GreedyStrategy


class ValueRerankingAgent:
    """
    Wrapper that adds value reranking to an existing AMAGO agent.

    This wrapper intercepts the agent's forward pass and modifies action selection
    to use top-k value reranking instead of stochastic sampling.

    Usage:
        # Create base agent
        base_agent = pretrained_model.initialize_agent(checkpoint=40)

        # Wrap with value reranking
        agent = ValueRerankingAgent(base_agent, k=5, aggregation='mean')

        # Use normally in evaluation
        results = agent.evaluate_test(...)
    """

    def __init__(
        self,
        base_agent,
        k: int = 5,
        aggregation: Literal["mean", "min", "max"] = "mean",
        use_policy_prior: bool = True,
        fallback_to_greedy: bool = True,
    ):
        """
        Args:
            base_agent: Pretrained AMAGO agent (actor-critic)
            k: Number of top actions to consider from policy
            aggregation: How to aggregate critic ensemble ('mean', 'min', 'max')
            use_policy_prior: Weight values by policy probability
            fallback_to_greedy: Fallback to greedy if value reranking fails
        """
        self.base_agent = base_agent
        self.k = k
        self.aggregation = aggregation
        self.use_policy_prior = use_policy_prior
        self.fallback_to_greedy = fallback_to_greedy

        # Verify agent has critic
        if not hasattr(base_agent, 'critic'):
            raise ValueError(
                "Base agent must have 'critic' attribute. "
                "Make sure you're using an RL agent (not IL-only)."
            )

        # Create strategy
        self.strategy = ValueRerankingStrategy(
            agent=base_agent,
            k=k,
            aggregation=aggregation,
            use_policy_prior=use_policy_prior,
            fallback_to_greedy=fallback_to_greedy,
        )

        print(f"ValueRerankingAgent initialized:")
        print(f"  k={k}, aggregation={aggregation}")
        print(f"  use_policy_prior={use_policy_prior}")
        print(f"  fallback_to_greedy={fallback_to_greedy}")

    def __getattr__(self, name):
        """Delegate all other attributes to base agent."""
        return getattr(self.base_agent, name)

    def act(self, obs, *args, **kwargs):
        """
        Modified act() that uses value reranking.

        This intercepts the agent's action selection and applies value reranking
        to the top-k actions from the policy.
        """
        # Call base agent's internal forward pass to get logits and state
        # This is the tricky part - we need to hook into AMAGO's internals

        # AMAGO agents typically have:
        # 1. self.base_agent.policy - the full policy network
        # 2. Hidden state management for recurrent encoders
        # 3. Actor outputs logits

        # We need to:
        # 1. Get logits from actor (via policy forward pass)
        # 2. Get state embedding for critic
        # 3. Apply value reranking
        # 4. Return selected action

        try:
            return self._value_reranked_act(obs, *args, **kwargs)
        except Exception as e:
            if self.fallback_to_greedy:
                warnings.warn(
                    f"Value reranking failed in act(): {e}\n"
                    f"Falling back to base agent's act(). "
                    f"This might happen if critic requires different inputs."
                )
                return self.base_agent.act(obs, *args, **kwargs)
            else:
                raise

    def _value_reranked_act(self, obs, *args, **kwargs):
        """
        Core value reranking logic for act().

        This method needs to be adapted based on AMAGO's actual API.
        The exact implementation depends on how AMAGO exposes:
        - Policy forward pass
        - State embeddings
        - Critic evaluation

        TODO: This is a template that needs to be filled in based on
        actual AMAGO agent structure. See test_critic_access.py results.
        """
        # Placeholder implementation
        # Real implementation will depend on AMAGO's API

        # Step 1: Process observation through encoders
        # processed_obs = self.base_agent.process_obs(obs)

        # Step 2: Get state embedding
        # state_embedding = self.base_agent.get_state_embedding(processed_obs)

        # Step 3: Get actor logits
        # logits = self.base_agent.actor(state_embedding, obs)

        # Step 4: Get illegal action mask (if available in obs)
        # illegal_mask = obs.get('illegal_actions', None)

        # Step 5: Apply value reranking
        # action = self.strategy.select_action(logits, illegal_mask, state_embedding)

        # For now, fall back to base agent
        raise NotImplementedError(
            "Value reranking requires understanding AMAGO's internal API. "
            "Run test_critic_access.py first to understand agent structure, "
            "then implement _value_reranked_act() accordingly."
        )


def wrap_agent_with_value_reranking(
    base_agent,
    k: int = 5,
    aggregation: Literal["mean", "min", "max"] = "mean",
    use_policy_prior: bool = True,
):
    """
    Convenience function to wrap an agent with value reranking.

    Args:
        base_agent: AMAGO agent to wrap
        k: Number of top actions to consider
        aggregation: Critic ensemble aggregation method
        use_policy_prior: Whether to weight values by policy

    Returns:
        Wrapped agent with value reranking

    Example:
        >>> from metamon.rl.pretrained import SmallRL
        >>> from metamon.rl.value_reranking_wrapper import wrap_agent_with_value_reranking
        >>>
        >>> model = SmallRL()
        >>> base_agent = model.initialize_agent(checkpoint=40)
        >>> agent = wrap_agent_with_value_reranking(base_agent, k=5, aggregation='mean')
        >>>
        >>> # Use in evaluation
        >>> results = agent.evaluate_test(...)
    """
    return ValueRerankingAgent(
        base_agent=base_agent,
        k=k,
        aggregation=aggregation,
        use_policy_prior=use_policy_prior,
    )


# Alternative: Simpler greedy wrapper (no critic needed)
class GreedyAgent:
    """
    Simple wrapper that uses greedy action selection (argmax from policy).

    This is simpler than value reranking and doesn't require critic access.
    Good baseline to test if deterministic selection helps.
    """

    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.strategy = GreedyStrategy(base_agent)
        print("GreedyAgent initialized (deterministic policy)")

    def __getattr__(self, name):
        """Delegate all other attributes to base agent."""
        return getattr(self.base_agent, name)

    def act(self, obs, *args, **kwargs):
        """
        Act greedily (argmax from policy).

        Note: This is a simplified implementation.
        Real implementation needs to hook into AMAGO's action selection.
        """
        # For now, this is a placeholder
        # Real implementation requires understanding AMAGO's act() internals
        warnings.warn(
            "GreedyAgent.act() is not fully implemented yet. "
            "Using base agent's act(). "
            "Implement after understanding AMAGO API from test_critic_access.py"
        )
        return self.base_agent.act(obs, *args, **kwargs)
