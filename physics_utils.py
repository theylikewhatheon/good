"""Physics calculations and prediction for the Nexto bot."""

import math
from .vector_utils import Vec3


class PhysicsConstants:
    """Game physics constants."""
    GRAVITY = 650.0
    AIR_DRAG_COEFFICIENT = 0.0305
    GROUND_FRICTION = 0.99
    BOOST_ACCELERATION = 991.666
    THROTTLE_ACCELERATION = 1600.0
    SUPERSONIC_SPEED = 2300.0
    GROUND_HEIGHT = 17.01
    ARENA_CEILING = 2044.0
    ARENA_WIDTH = 4096.0
    ARENA_LENGTH = 5120.0
    PHYSICS_TICK_RATE = 120.0


class MotionPredictor:
    """Handles motion prediction for cars."""
    
    def __init__(self):
        self.predicted_path = []
    
    def predict_target_motion(self, car, steps=180):
        """
        Enhanced prediction with acceleration and advanced physics modeling.
        
        Args:
            car: The car object to predict motion for
            steps: Number of prediction steps (default 180 = 1.5 seconds at 120Hz)
        
        Returns:
            List of Vec3 positions representing the predicted path
        """
        self.predicted_path = []
        pos = Vec3.from_rlbot_vector(car.physics.location)
        vel = Vec3.from_rlbot_vector(car.physics.velocity)
        ang_vel = Vec3.from_rlbot_vector(car.physics.angular_velocity)
        
        # Physics constants
        time_step = 1 / PhysicsConstants.PHYSICS_TICK_RATE
        
        # Get car orientation for thrust calculations
        car_forward = Vec3(
            math.cos(car.physics.rotation.yaw) * math.cos(car.physics.rotation.pitch),
            math.sin(car.physics.rotation.yaw) * math.cos(car.physics.rotation.pitch),
            math.sin(car.physics.rotation.pitch)
        )
        
        for i in range(steps):
            # Store position
            self.predicted_path.append(Vec3(pos.x, pos.y, pos.z))
            
            # Update position
            pos = pos + vel * time_step
            
            if car.has_wheel_contact:
                # Ground physics
                self._apply_ground_physics(vel, ang_vel, car_forward, car, time_step)
                
                # Keep grounded
                pos.z = PhysicsConstants.GROUND_HEIGHT
                vel.z = 0
            else:
                # Aerial physics
                self._apply_aerial_physics(pos, vel, time_step)
                
                # Check for ground collision
                if pos.z <= PhysicsConstants.GROUND_HEIGHT and vel.z <= 0:
                    pos.z = PhysicsConstants.GROUND_HEIGHT
                    vel.z = 0
                    car.has_wheel_contact = True
            
            # Apply arena boundaries
            self._apply_arena_boundaries(pos, vel)
        
        return self.predicted_path
    
    def _apply_ground_physics(self, vel, ang_vel, car_forward, car, time_step):
        """Apply ground-based physics calculations."""
        # Apply turning
        if abs(ang_vel.z) > 0.01:
            speed = vel.magnitude()
            if speed > 100:
                # Rotate velocity vector
                yaw_change = ang_vel.z * time_step
                cos_yaw = math.cos(yaw_change)
                sin_yaw = math.sin(yaw_change)
                new_vel_x = vel.x * cos_yaw - vel.y * sin_yaw
                new_vel_y = vel.x * sin_yaw + vel.y * cos_yaw
                vel.x = new_vel_x
                vel.y = new_vel_y
        
        # Apply ground friction
        vel.x *= PhysicsConstants.GROUND_FRICTION
        vel.y *= PhysicsConstants.GROUND_FRICTION
        
        # Estimate acceleration/deceleration
        if car.boost > 0:
            # Assume they might boost
            accel_vector = car_forward * PhysicsConstants.BOOST_ACCELERATION * time_step
            vel = vel + accel_vector
            # Cap at supersonic
            if vel.magnitude() > PhysicsConstants.SUPERSONIC_SPEED:
                vel = vel.normalized() * PhysicsConstants.SUPERSONIC_SPEED
    
    def _apply_aerial_physics(self, pos, vel, time_step):
        """Apply aerial physics calculations."""
        # Gravity
        vel.z -= PhysicsConstants.GRAVITY * time_step
        
        # Air drag
        drag_force = PhysicsConstants.AIR_DRAG_COEFFICIENT * vel.magnitude()
        if vel.magnitude() > 0:
            vel = vel - (vel.normalized() * drag_force * time_step)
        
        # Arena ceiling
        if pos.z >= PhysicsConstants.ARENA_CEILING:
            pos.z = PhysicsConstants.ARENA_CEILING
            vel.z = -vel.z * 0.6  # Bounce with damping
    
    def _apply_arena_boundaries(self, pos, vel):
        """Apply arena wall boundaries."""
        # Arena walls (simplified)
        if abs(pos.x) >= PhysicsConstants.ARENA_WIDTH:
            pos.x = PhysicsConstants.ARENA_WIDTH * (1 if pos.x > 0 else -1)
            vel.x = -vel.x * 0.6
        if abs(pos.y) >= PhysicsConstants.ARENA_LENGTH:
            pos.y = PhysicsConstants.ARENA_LENGTH * (1 if pos.y > 0 else -1)
            vel.y = -vel.y * 0.6


class InterceptPlanner:
    """Handles intercept point calculations."""
    
    def plan_intercept(self, my_car, predicted_path):
        """
        Enhanced intercept planning with acceleration modeling.
        
        Args:
            my_car: The bot's car object
            predicted_path: List of predicted positions
        
        Returns:
            Vec3 intercept point or None if no valid intercept found
        """
        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        my_vel = Vec3.from_rlbot_vector(my_car.physics.velocity)
        my_boost = my_car.boost
        
        intercept_point = None
        best_intercept_score = float('-inf')
        
        for i, future_pos in enumerate(predicted_path):
            time_in_future = (i + 1) * (1 / PhysicsConstants.PHYSICS_TICK_RATE)
            
            # Vector to intercept point
            intercept_vector = future_pos - my_pos
            intercept_distance = intercept_vector.magnitude()
            
            if intercept_distance < 1:
                continue
                
            intercept_direction = intercept_vector.normalized()
            
            # Calculate if we can reach this position
            if self._can_reach_position(my_vel, my_boost, intercept_direction, 
                                      intercept_distance, time_in_future):
                # Calculate intercept quality score
                score = self._calculate_intercept_score(my_vel, intercept_direction, 
                                                      time_in_future, my_boost)
                
                if score > best_intercept_score:
                    best_intercept_score = score
                    intercept_point = future_pos
        
        return intercept_point
    
    def _can_reach_position(self, my_vel, my_boost, direction, distance, time):
        """Check if we can reach a position in the given time."""
        current_speed_toward = my_vel.dot(direction)
        boost_time_available = min(my_boost / 33.3, time)
        no_boost_time = time - boost_time_available
        
        # Kinematic equation for reachable distance
        reachable_distance = (
            current_speed_toward * time +
            0.5 * PhysicsConstants.BOOST_ACCELERATION * boost_time_available**2 +
            0.5 * PhysicsConstants.THROTTLE_ACCELERATION * no_boost_time**2 +
            PhysicsConstants.BOOST_ACCELERATION * boost_time_available * no_boost_time
        )
        
        return reachable_distance >= distance
    
    def _calculate_intercept_score(self, my_vel, direction, time_in_future, my_boost):
        """Calculate quality score for an intercept point."""
        alignment = my_vel.normalized().dot(direction) if my_vel.magnitude() > 100 else 0
        time_efficiency = 1.0 / (1.0 + time_in_future)  # Prefer sooner intercepts
        boost_time_available = min(my_boost / 33.3, time_in_future)
        current_speed_toward = my_vel.dot(direction)
        speed_score = min(PhysicsConstants.SUPERSONIC_SPEED, 
                         current_speed_toward + PhysicsConstants.BOOST_ACCELERATION * boost_time_available) / PhysicsConstants.SUPERSONIC_SPEED
        
        return alignment * 0.3 + time_efficiency * 0.5 + speed_score * 0.2