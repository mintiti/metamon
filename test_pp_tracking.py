#!/usr/bin/env python
"""
Simple test script for PPTrackingObservationSpace.

Tests that the new observation space correctly tracks PP for all Pokemon.
"""
import numpy as np
from metamon.interface import (
    PPTrackingObservationSpace,
    UniversalState,
    UniversalPokemon,
    UniversalMove,
)


def create_mock_pokemon(name: str, num_moves: int = 4, pp_values=None):
    """Create a mock Pokemon with specified moves and PP values."""
    if pp_values is None:
        pp_values = [(32, 32), (24, 24), (16, 16), (8, 8)]

    moves = []
    for i in range(num_moves):
        current_pp, max_pp = pp_values[i] if i < len(pp_values) else (10, 10)
        move = UniversalMove(
            name=f"move{i}",
            move_type="normal",
            category="physical",
            base_power=80,
            accuracy=1.0,
            priority=0,
            current_pp=current_pp,
            max_pp=max_pp,
        )
        moves.append(move)

    return UniversalPokemon(
        name=name,
        base_species=name,
        hp_pct=1.0,
        types="normal",
        item="noitem",
        ability="noability",
        lvl=100,
        status="nostatus",
        effect="noeffect",
        moves=moves,
        atk_boost=0,
        spa_boost=0,
        def_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        base_atk=100,
        base_spa=100,
        base_def=100,
        base_spd=100,
        base_spe=100,
        base_hp=100,
        tera_type="notype",
    )


def create_mock_state(
    player_active_name="pikachu",
    opponent_active_name="charizard",
    num_switches=2,
):
    """Create a mock UniversalState for testing."""
    player_active = create_mock_pokemon(player_active_name, 4, [(32, 32), (24, 24), (16, 16), (8, 8)])
    opponent_active = create_mock_pokemon(opponent_active_name, 4, [(30, 32), (20, 24), (12, 16), (4, 8)])

    # Create some switches (bench Pokemon)
    switches = []
    for i in range(num_switches):
        switches.append(create_mock_pokemon(f"switch{i}", 3, [(25, 32), (15, 24), (10, 16)]))

    return UniversalState(
        format="gen1ou",
        player_active_pokemon=player_active,
        opponent_active_pokemon=opponent_active,
        available_switches=switches,
        player_prev_move=UniversalMove.blank_move(),
        opponent_prev_move=UniversalMove.blank_move(),
        opponents_remaining=6,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=False,
        opponent_teampreview=[],
    )


def test_pp_tracking_basic():
    """Test basic PP tracking functionality."""
    print("=" * 60)
    print("Testing PPTrackingObservationSpace")
    print("=" * 60)

    # Create observation space
    obs_space = PPTrackingObservationSpace()
    obs_space.reset()

    # Create first state
    state1 = create_mock_state("pikachu", "charizard", num_switches=2)
    obs1 = obs_space.state_to_obs(state1)

    print("\n1. First observation:")
    print(f"   Observation keys: {obs1.keys()}")
    print(f"   Numbers shape: {obs1['numbers'].shape}")
    print(f"   Expected shape: (103,) [55 from ExpandedObservationSpace + 48 from PP tracking]")

    # Check shape
    assert obs1['numbers'].shape == (103,), f"Expected (103,), got {obs1['numbers'].shape}"
    print("   ✓ Shape is correct!")

    # Extract PP features (last 48 values)
    pp_features = obs1['numbers'][-48:]
    print(f"\n2. PP features (last 48 values):")
    print(f"   Player PP features (first 24): {pp_features[:24]}")
    print(f"   Opponent PP features (last 24): {pp_features[24:]}")

    # Check that we have tracked Pokemon
    print(f"\n3. Tracked Pokemon:")
    print(f"   Player order: {obs_space.player_pokemon_order}")
    print(f"   Opponent order: {obs_space.opponent_pokemon_order}")
    print(f"   Player PP dict: {obs_space.player_pokemon_pp}")
    print(f"   Opponent PP dict: {obs_space.opponent_pokemon_pp}")

    # Verify PP ratios for player's first Pokemon (pikachu)
    # Expected: [32/32=1.0, 24/24=1.0, 16/16=1.0, 8/8=1.0]
    player_first_4_moves = pp_features[0:4]
    print(f"\n4. Player's active Pokemon (pikachu) PP ratios:")
    print(f"   {player_first_4_moves}")
    print(f"   Expected: [1.0, 1.0, 1.0, 1.0] (all moves at full PP)")
    assert np.allclose(player_first_4_moves, [1.0, 1.0, 1.0, 1.0]), "Player active Pokemon PP incorrect!"
    print("   ✓ Correct!")

    # Verify opponent's PP ratios (charizard)
    # Expected: [30/32≈0.94, 20/24≈0.83, 12/16=0.75, 4/8=0.5]
    opponent_first_4_moves = pp_features[24:28]
    expected_opponent = [30/32, 20/24, 12/16, 4/8]
    print(f"\n5. Opponent's active Pokemon (charizard) PP ratios:")
    print(f"   {opponent_first_4_moves}")
    print(f"   Expected: {expected_opponent}")
    assert np.allclose(opponent_first_4_moves, expected_opponent, atol=0.01), "Opponent active Pokemon PP incorrect!"
    print("   ✓ Correct!")

    # Test with a second state (opponent switches)
    print("\n" + "=" * 60)
    print("Testing with opponent switch")
    print("=" * 60)

    state2 = create_mock_state("pikachu", "blastoise", num_switches=2)
    obs2 = obs_space.state_to_obs(state2)

    print(f"\n6. After opponent switches:")
    print(f"   Opponent order: {obs_space.opponent_pokemon_order}")
    print(f"   Should now have: ['charizard', 'blastoise']")
    assert len(obs_space.opponent_pokemon_order) == 2, "Should have tracked 2 opponent Pokemon!"
    print("   ✓ Correctly tracking multiple opponent Pokemon!")

    # Check that unseen Pokemon slots are -1.0
    pp_features2 = obs2['numbers'][-48:]
    print(f"\n7. Checking unseen Pokemon slots:")
    # Player should have 3 tracked (pikachu + 2 switches), so slots 4-6 should be -1.0
    # Each slot has 4 moves, so indices 12-24 (3*4 to 6*4) should be -1.0
    player_unseen = pp_features2[12:24]
    print(f"   Player unseen slots (Pokemon 4-6): {player_unseen}")
    print(f"   Expected: all -1.0")
    assert np.allclose(player_unseen, -1.0), "Unseen player slots should be -1.0!"
    print("   ✓ Correct!")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_pp_tracking_reset():
    """Test that reset() properly clears tracking."""
    print("\n" + "=" * 60)
    print("Testing reset() functionality")
    print("=" * 60)

    obs_space = PPTrackingObservationSpace()
    obs_space.reset()

    # Generate an observation
    state = create_mock_state("pikachu", "charizard", num_switches=1)
    obs_space.state_to_obs(state)

    print(f"\n1. Before reset:")
    print(f"   Player Pokemon tracked: {len(obs_space.player_pokemon_order)}")
    print(f"   Opponent Pokemon tracked: {len(obs_space.opponent_pokemon_order)}")

    # Reset and check
    obs_space.reset()

    print(f"\n2. After reset:")
    print(f"   Player Pokemon tracked: {len(obs_space.player_pokemon_order)}")
    print(f"   Opponent Pokemon tracked: {len(obs_space.opponent_pokemon_order)}")

    assert len(obs_space.player_pokemon_order) == 0, "Player order should be empty after reset!"
    assert len(obs_space.opponent_pokemon_order) == 0, "Opponent order should be empty after reset!"
    print("   ✓ Reset works correctly!")


if __name__ == "__main__":
    test_pp_tracking_basic()
    test_pp_tracking_reset()
    print("\n🎉 All tests completed successfully!")
