"""Kickoff logic for the Nexto bot."""

import numpy as np
from rlbot.agents.base_agent import SimpleControllerState


# Predefined kickoff sequence
KICKOFF_CONTROLS = (
    11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
    + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
    + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
    + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
    for scs in KICKOFF_CONTROLS
])


class KickoffController:
    """Handles kickoff logic."""
    
    def __init__(self):
        self.kickoff_index = -1
    
    def update(self, packet, action):
        """
        Update kickoff behavior.
        
        Args:
            packet: GameTickPacket
            action: Current action array
        
        Returns:
            Modified action array if in kickoff, otherwise original action
        """
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index == -1:
                self.kickoff_index = 0
            
            if self.kickoff_index < len(KICKOFF_NUMPY):
                action = KICKOFF_NUMPY[self.kickoff_index].copy()
                self.kickoff_index += 1
        else:
            self.kickoff_index = -1
        
        return action