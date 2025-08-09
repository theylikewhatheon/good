"""Placeholder observation builder for Nexto bot."""

import numpy as np

# Placeholder boost locations
BOOST_LOCATIONS = [
    np.array([0, 0, 0]),  # Placeholder boost pad locations
]


class NextoObsBuilder:
    """Placeholder observation builder."""
    
    def __init__(self, field_info=None):
        self.field_info = field_info
    
    def build_obs(self, player, game_state, action):
        """
        Build observation from game state.
        
        Args:
            player: Player object
            game_state: GameState object
            action: Previous action
        
        Returns:
            np.array: Observation array
        """
        # Placeholder implementation - would normally build complex observation
        return np.zeros(100)  # Dummy observation