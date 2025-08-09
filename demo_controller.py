"""Demo logic for the Nexto bot."""

import math
import numpy as np
from .vector_utils import Vec3
from .bot_enums import DemolishPhase
from .physics_utils import MotionPredictor, InterceptPlanner


class DemoController:
    """Handles all demolition-related logic."""
    
    def __init__(self):
        self.demo_phase = DemolishPhase.CHOOSE_TARGET
        self.demo_target_index = -1
        self.demo_target_car = None
        self.demo_intercept_point = None
        
        # Dodge state
        self.is_dodging = False
        self.dodge_start_time = 0.0
        self.dodge_action = np.zeros(8)
        
        # Utility classes
        self.motion_predictor = MotionPredictor()
        self.intercept_planner = InterceptPlanner()
    
    def update(self, packet, my_car_index):
        """
        Main update loop for demo mode.
        
        Args:
            packet: GameTickPacket
            my_car_index: Index of our car
        
        Returns:
            np.ndarray: Action array
        """
        my_car = packet.game_cars[my_car_index]
        action = np.zeros(8)

        if self.is_dodging:
            return self._continue_dodge(packet)

        if my_car.is_demolished:
            self.demo_phase = DemolishPhase.CHOOSE_TARGET
            return action  # Empty controls

        if self.demo_phase == DemolishPhase.CHOOSE_TARGET:
            self._choose_target(packet, my_car_index)
        
        elif self.demo_phase == DemolishPhase.CHASING:
            if not self._is_target_valid(packet):
                self.demo_phase = DemolishPhase.CHOOSE_TARGET
            else:
                self._update_target_prediction(packet)
                self._update_intercept_point(my_car)
                action = self._execute_chase(my_car, packet)
        
        elif self.demo_phase == DemolishPhase.FAILED:
            action[0] = 1.0  # Drive forward if no targets

        return action
    
    def _choose_target(self, packet, my_car_index):
        """Select the best enemy to demolish."""
        my_car = packet.game_cars[my_car_index]
        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        best_target_index = -1
        min_dist = float('inf')

        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            if i == my_car_index or car.team == my_car.team or car.is_demolished:
                continue

            dist = my_pos.distance(Vec3.from_rlbot_vector(car.physics.location))
            if dist < min_dist:
                min_dist = dist
                best_target_index = i

        if best_target_index != -1:
            self.demo_target_index = best_target_index
            self.demo_target_car = packet.game_cars[self.demo_target_index]
            self.demo_phase = DemolishPhase.CHASING
        else:
            self.demo_target_car = None
            self.demo_phase = DemolishPhase.FAILED
    
    def _is_target_valid(self, packet):
        """Check if current target is still valid."""
        if self.demo_target_index == -1 or self.demo_target_index >= packet.num_cars:
            return False
        target = packet.game_cars[self.demo_target_index]
        return not target.is_demolished and target.team != packet.game_cars[0].team
    
    def _update_target_prediction(self, packet):
        """Update the predicted path of the target."""
        self.demo_target_car = packet.game_cars[self.demo_target_index]
        self.motion_predictor.predict_target_motion(self.demo_target_car)
    
    def _update_intercept_point(self, my_car):
        """Update the intercept point calculation."""
        self.demo_intercept_point = self.intercept_planner.plan_intercept(
            my_car, self.motion_predictor.predicted_path
        )
    
    def _execute_chase(self, my_car, packet):
        """Execute the chase behavior."""
        action = np.zeros(8)

        if self.demo_target_car is None or self.demo_target_index < 0:
            action[0] = 1.0  # Default throttle forward
            return action

        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        target_pos = Vec3.from_rlbot_vector(self.demo_target_car.physics.location)
        
        # Use intercept point if available, otherwise target current position
        chase_target = self.demo_intercept_point if self.demo_intercept_point else target_pos
        
        # Calculate steering and throttle
        action = self._calculate_chase_controls(my_car, chase_target)
        
        # Check if we should dodge
        distance = my_pos.distance(chase_target)
        if distance < 200 and not self.is_dodging:
            self._initiate_dodge(packet, action)
        
        return action
    
    def _calculate_chase_controls(self, my_car, target_pos):
        """Calculate steering and throttle for chasing."""
        action = np.zeros(8)
        
        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        direction = (target_pos - my_pos).normalized()
        
        # Calculate car's forward direction
        my_yaw = my_car.physics.rotation.yaw
        car_forward = Vec3(math.cos(my_yaw), math.sin(my_yaw), 0)
        
        # Calculate steering angle
        cross_product = car_forward.x * direction.y - car_forward.y * direction.x
        
        # Set controls
        action[1] = np.clip(cross_product * 3.0, -1.0, 1.0)  # Steer
        action[0] = 1.0  # Full throttle
        action[6] = 1.0 if my_car.boost > 0 else 0.0  # Boost if available
        
        return action
    
    def _initiate_dodge(self, packet, action):
        """Initiate a dodge maneuver."""
        self.is_dodging = True
        self.dodge_start_time = packet.game_info.seconds_elapsed
        self.dodge_action = action.copy()
        action[5] = 1.0  # Jump
    
    def _continue_dodge(self, packet):
        """Continue executing a dodge maneuver."""
        time_since_dodge_start = packet.game_info.seconds_elapsed - self.dodge_start_time
        
        # Start with the saved action plan
        action = self.dodge_action.copy()
        action[0] = 1.0  # Always throttle
        action[6] = 1.0  # Always boost during the dodge

        # Stage 1: The second jump to initiate the flip
        if 0.05 < time_since_dodge_start < 0.15:
            action[5] = 1.0  # Second jump
        else:
            action[5] = 0.0  # Ensure we don't jump after this window

        # Stage 2: The dodge is complete
        if time_since_dodge_start >= 0.5:
            self.is_dodging = False
        
        return action