"""
Inference Strategies for Metamon RL Agents

Provides different action selection strategies beyond the default stochastic sampling:
- Greedy: Argmax from policy (deterministic)
- Temperature: Temperature-scaled sampling
- ValueReranking: Top-k from policy, rerank by critic value (requires actor-critic agent)

These strategies improve inference-time performance without requiring retraining.
"""
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Literal
import warnings


class InferenceStrategy:
    """Base class for inference strategies."""

    def __init__(self, agent):
        """
        Args:
            agent: AMAGO agent with actor (and optionally critic)
        """
        self.agent = agent

    def select_action(self, logits: torch.Tensor, illegal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Select action from logits.

        Args:
            logits: Raw logits from policy, shape [batch, actions]
            illegal_mask: Boolean mask where True = illegal action, shape [batch, actions]

        Returns:
            action: Selected action indices, shape [batch]
        """
        raise NotImplementedError


class StochasticStrategy(InferenceStrategy):
    """Default stochastic sampling (baseline)."""

    def select_action(self, logits: torch.Tensor, illegal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if illegal_mask is not None:
            logits = logits.masked_fill(illegal_mask, -float("inf"))
        dist = Categorical(logits=logits)
        return dist.sample()


class GreedyStrategy(InferenceStrategy):
    """Greedy action selection (argmax from policy)."""

    def select_action(self, logits: torch.Tensor, illegal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if illegal_mask is not None:
            logits = logits.masked_fill(illegal_mask, -float("inf"))
        return logits.argmax(dim=-1)


class TemperatureStrategy(InferenceStrategy):
    """Temperature-scaled sampling."""

    def __init__(self, agent, temperature: float = 1.0):
        """
        Args:
            agent: AMAGO agent
            temperature: Temperature parameter
                - temperature < 1.0: More deterministic (sharper distribution)
                - temperature = 1.0: Standard sampling
                - temperature > 1.0: More exploratory (flatter distribution)
                - temperature → 0: Greedy
        """
        super().__init__(agent)
        self.temperature = temperature

    def select_action(self, logits: torch.Tensor, illegal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if illegal_mask is not None:
            logits = logits.masked_fill(illegal_mask, -float("inf"))

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        if self.temperature < 0.01:
            # Effectively greedy
            return scaled_logits.argmax(dim=-1)
        else:
            dist = Categorical(logits=scaled_logits)
            return dist.sample()


class ValueRerankingStrategy(InferenceStrategy):
    """
    Top-k value reranking strategy.

    1. Sample top-k actions from policy (actor)
    2. Evaluate each with value function (critic)
    3. Select action with highest value

    Combines policy and value function for better decision-making.
    Similar to AlphaGo's policy+value approach.
    """

    def __init__(
        self,
        agent,
        k: int = 5,
        aggregation: Literal["mean", "min", "max"] = "mean",
        use_policy_prior: bool = True,
        fallback_to_greedy: bool = True,
    ):
        """
        Args:
            agent: AMAGO agent with actor and critic
            k: Number of top actions to consider from policy
            aggregation: How to aggregate multi-critic ensemble
                - 'mean': Average of all critics (balanced)
                - 'min': Minimum across critics (conservative/pessimistic)
                - 'max': Maximum across critics (optimistic)
            use_policy_prior: If True, weight values by policy probability
            fallback_to_greedy: If critic fails, fallback to greedy policy
        """
        super().__init__(agent)
        self.k = k
        self.aggregation = aggregation
        self.use_policy_prior = use_policy_prior
        self.fallback_to_greedy = fallback_to_greedy

        # Verify critic exists
        if not hasattr(agent, 'critic'):
            raise ValueError(
                "Agent must have 'critic' attribute for ValueRerankingStrategy. "
                "Make sure the agent is actor-critic (not IL-only)."
            )

    def select_action(
        self,
        logits: torch.Tensor,
        illegal_mask: Optional[torch.Tensor] = None,
        state_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select action using top-k value reranking.

        Args:
            logits: Raw logits from policy, shape [batch, actions]
            illegal_mask: Boolean mask where True = illegal action
            state_embedding: State embedding for critic evaluation
                If None, will try to use agent's current hidden state

        Returns:
            action: Selected action indices, shape [batch]
        """
        try:
            return self._value_rerank(logits, illegal_mask, state_embedding)
        except Exception as e:
            if self.fallback_to_greedy:
                warnings.warn(
                    f"ValueReranking failed ({e}), falling back to greedy. "
                    f"Consider using GreedyStrategy directly."
                )
                return GreedyStrategy(self.agent).select_action(logits, illegal_mask)
            else:
                raise

    def _value_rerank(
        self,
        logits: torch.Tensor,
        illegal_mask: Optional[torch.Tensor],
        state_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Core value reranking logic."""
        batch_size, num_actions = logits.shape

        # Apply illegal action mask
        if illegal_mask is not None:
            masked_logits = logits.masked_fill(illegal_mask, -float("inf"))
        else:
            masked_logits = logits

        # Get top-k actions from policy
        # Clamp k to not exceed legal actions
        legal_actions_per_batch = (~illegal_mask).sum(dim=-1) if illegal_mask is not None else torch.full((batch_size,), num_actions)
        k_clamped = min(self.k, legal_actions_per_batch.min().item())

        if k_clamped < 1:
            raise ValueError("No legal actions available")

        # Get top-k policy probabilities and indices
        top_k_logits, top_k_indices = masked_logits.topk(k=k_clamped, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # Shape: [batch, k]

        # Evaluate each top-k action with critic
        if state_embedding is None:
            # Try to get state from agent's current computation
            # This requires the agent to expose its hidden state
            raise NotImplementedError(
                "state_embedding is required. Pass the state embedding from "
                "the agent's trajectory encoder output."
            )

        # Get Q-values from critic for all actions
        # Critic output shape depends on implementation:
        # - Option A: [batch, num_critics, num_actions]
        # - Option B: [batch, num_actions]
        with torch.no_grad():
            q_values = self.agent.critic(state_embedding)  # Forward pass

            # Handle different critic output formats
            if q_values.dim() == 3:
                # Multi-critic ensemble: [batch, num_critics, num_actions]
                # Aggregate across critics
                if self.aggregation == "mean":
                    q_values = q_values.mean(dim=1)  # [batch, num_actions]
                elif self.aggregation == "min":
                    q_values = q_values.min(dim=1)[0]  # [batch, num_actions]
                elif self.aggregation == "max":
                    q_values = q_values.max(dim=1)[0]  # [batch, num_actions]
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")
            elif q_values.dim() != 2:
                raise ValueError(f"Unexpected critic output shape: {q_values.shape}")

            # Extract Q-values for top-k actions
            # top_k_indices: [batch, k]
            # q_values: [batch, num_actions]
            # Result: [batch, k]
            top_k_q_values = torch.gather(q_values, dim=-1, index=top_k_indices)

        # Combine policy and value
        if self.use_policy_prior:
            # Weight Q-values by policy probability (policy-guided value)
            # This helps when policy is confident but value is uncertain
            combined_scores = top_k_probs * top_k_q_values
        else:
            # Pure value-based selection
            combined_scores = top_k_q_values

        # Select action with highest combined score
        best_k_idx = combined_scores.argmax(dim=-1)  # [batch]

        # Map back to original action index
        selected_actions = top_k_indices.gather(dim=-1, index=best_k_idx.unsqueeze(-1)).squeeze(-1)

        return selected_actions


# Convenience factory function
def get_inference_strategy(
    agent,
    strategy: str = "stochastic",
    **kwargs
) -> InferenceStrategy:
    """
    Factory function to create inference strategies.

    Args:
        agent: AMAGO agent
        strategy: Strategy name
            - 'stochastic': Default sampling (baseline)
            - 'greedy': Argmax from policy
            - 'temperature': Temperature-scaled sampling
            - 'value_rerank': Top-k value reranking
        **kwargs: Strategy-specific parameters

    Returns:
        InferenceStrategy instance

    Examples:
        >>> strategy = get_inference_strategy(agent, 'greedy')
        >>> strategy = get_inference_strategy(agent, 'temperature', temperature=0.5)
        >>> strategy = get_inference_strategy(agent, 'value_rerank', k=5, aggregation='min')
    """
    strategies = {
        "stochastic": StochasticStrategy,
        "greedy": GreedyStrategy,
        "temperature": TemperatureStrategy,
        "value_rerank": ValueRerankingStrategy,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy](agent, **kwargs)
