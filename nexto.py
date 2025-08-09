import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.quick_chats import QuickChats
from rlgym_compat import GameState
import math
import random
from enum import Enum, auto
import keyboard
from .agent import Agent
from .nexto_obs import NextoObsBuilder, BOOST_LOCATIONS

# --- Helper Classes for Demo Logic ---

class Vec3:
    """A minimal vector class to mirror the logic from the Kotlin bot."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def from_rlbot_vector(vec):
        return Vec3(vec.x, vec.y, vec.z) if vec is not None else Vec3()

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self):
        mag = self.magnitude()
        return self if mag == 0 else Vec3(self.x / mag, self.y / mag, self.z / mag)

    def distance(self, other):
        return (self - other).magnitude()

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

class BotMode(Enum):
    NORMAL = auto()
    DEMO = auto()

class DemolishPhase(Enum):
    """State machine for the advanced demo logic."""
    CHOOSE_TARGET = auto()
    CHASING = auto()
    FAILED = auto()

# --- End Helper Classes ---


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


class Nexto(BaseAgent):
    # Class-level flag to prevent multiple spams
    _has_printed_init = False

    def __init__(self, name, team, index,
                 beta=1, render=False, hardcoded_kickoffs=True, stochastic_kickoffs=True):
        super().__init__(name, team, index)

        self.obs_builder = None
        self.agent = Agent()
        self.tick_skip = 8

        # Beta controls randomness:
        # 1=best action, 0.5=sampling from probability, 0=random, -1=worst action, or anywhere inbetween
        self.beta = beta
        self.render = render
        self.hardcoded_kickoffs = hardcoded_kickoffs
        self.stochastic_kickoffs = stochastic_kickoffs

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.kickoff_index = -1
        self.field_info = None

        # --- Advanced Demo State ---
        self.mode = BotMode.NORMAL
        self.c_key_pressed_last_tick = False
        # State machine and target tracking
        self.demo_phase = DemolishPhase.CHOOSE_TARGET
        self.demo_target_index = -1
        self.demo_target_car = None
        # Prediction and Intercept
        self.demo_predicted_path = []
        self.demo_intercept_point = None
        # Dodge state
        self.is_dodging = False
        self.dodge_start_time = 0.0
        self.dodge_action = np.zeros(8)

        # toxic handling
        self.isToxic = False
        self.orangeGoals = 0
        self.blueGoals = 0
        self.demoedCount = 0
        self.lastFrameBall = None
        self.lastFrameDemod = False
        self.demoCount = 0
        self.pesterCount = 0
        self.demoedTickCount = 0
        self.demoCalloutCount = 0
        self.lastPacket = None

        # Only print these lines the *first time* this class is ever constructed
        if not Nexto._has_printed_init:
            print('Nexto Ready - Index:', index, 'Beta:', str(beta))
            print("Remember to run Nexto at 120fps with vsync off! "
                  "Stable 240/360 is second best if that's better for your eyes")
            print("Also check out the RLGym Twitch stream to watch live bot training and occasional showmatches!")
            Nexto._has_printed_init = True

    def initialize_agent(self, field_info):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.field_info = field_info
        self.obs_builder = NextoObsBuilder(field_info=self.field_info)
        self.game_state = GameState(self.field_info)
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.kickoff_index = -1

    def render_attention_weights(self, weights, positions, n=3):
        if weights is None:
            return
        mean_weights = torch.mean(torch.stack(weights), dim=0).numpy()[0][0]

        top = sorted(range(len(mean_weights)), key=lambda i: mean_weights[i], reverse=True)
        top.remove(0)  # Self

        self.renderer.begin_rendering('attention_weights')

        invert = np.array([-1, -1, 1]) if self.team == 1 else np.ones(3)
        loc = positions[0] * invert
        mx = mean_weights[~(np.arange(len(mean_weights)) == 1)].max()
        c = 1
        for i in top[:n]:
            weight = mean_weights[i] / mx
            dest = positions[i] * invert
            color = self.renderer.create_color(
                255, round(255 * (1 - weight)), round(255),
                round(255 * (1 - weight))
            )
            self.renderer.draw_string_3d(dest, 2, 2, str(c), color)
            c += 1
            self.renderer.draw_line_3d(loc, dest, color)
        self.renderer.end_rendering()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # --- Manual Mode Toggle ---
        c_key_is_down = keyboard.is_pressed('c')
        if c_key_is_down and not self.c_key_pressed_last_tick:
            if self.mode == BotMode.NORMAL:
                self.mode = BotMode.DEMO
                self.demo_phase = DemolishPhase.CHOOSE_TARGET # Reset demo state on enable
                self.send_quick_chat(QuickChats.CHAT_TEAM_ONLY, QuickChats.Custom_Useful_Bumping)
            else:
                self.mode = BotMode.NORMAL
                self.send_quick_chat(QuickChats.CHAT_TEAM_ONLY, QuickChats.Information_Defending)
        self.c_key_pressed_last_tick = c_key_is_down

        # --- ALWAYS UPDATE STATE ---
        # This is the critical fix. We must decode the packet every tick for both modes
        # to ensure the bot has fresh data, otherwise it thinks it has 0 boost.
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        # --- STATE-BASED LOGIC ROUTER ---
        if self.mode == BotMode.DEMO:
            # DEMO MODE: Run every tick, bypassing the Nexto tick_skip system.
            self.run_advanced_demo_logic_every_tick(packet)
        else: # BotMode.NORMAL
            # NEXTO MODE: Respect the standard tick_skip logic.
            if self.isToxic:
                self.toxicity(packet)

            self.run_nexto_logic(packet)

            if self.ticks >= self.tick_skip - 1:
                self.update_controls(self.action)

            if self.ticks >= self.tick_skip:
                self.ticks = 0
                self.update_action = True

            if self.hardcoded_kickoffs:
                self.maybe_do_kickoff(packet, ticks_elapsed)

        return self.controls

    def run_nexto_logic(self, packet: GameTickPacket):
        """The standard Nexto brain logic."""
        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)

            beta = self.beta
            # e.g., random if match ended
            if packet.game_info.is_match_ended:
                beta = 0
            # add some randomness on kickoffs
            if self.stochastic_kickoffs and packet.game_info.is_kickoff_pause:
                beta = 0.5

            self.action, weights = self.agent.act(obs, beta)

            if self.render:
                positions = np.asarray([p.car_data.position for p in self.game_state.players] +
                                       [self.game_state.ball.position] +
                                       list(BOOST_LOCATIONS))
                self.render_attention_weights(weights, positions)

    def run_advanced_demo_logic_every_tick(self, packet: GameTickPacket):
        """
        A dedicated logic loop for demo mode that runs every tick,
        ignoring the Nexto tick_skip system for maximum responsiveness.
        """
        my_car = packet.game_cars[self.index]
        action = np.zeros(8)

        if self.is_dodging:
            action = self.continue_demo_dodge(packet)
            self.update_controls(action)
            return

        if my_car.is_demolished:
            self.demo_phase = DemolishPhase.CHOOSE_TARGET
            self.update_controls(action) # Send empty controls
            return

        if self.demo_phase == DemolishPhase.CHOOSE_TARGET:
            self.choose_demo_target(packet)
            # action is already zeros, bot will do nothing until next tick
        
        elif self.demo_phase == DemolishPhase.CHASING:
            if not self.is_demo_target_valid(packet):
                self.demo_phase = DemolishPhase.CHOOSE_TARGET
            else:
                self.predict_target_motion(packet)
                self.plan_intercept(my_car)
                action = self.execute_chase(my_car, packet)
        
        elif self.demo_phase == DemolishPhase.FAILED:
            action[0] = 1.0 # Drive forward if no targets

        # In demo mode, we apply controls directly and immediately.
        self.update_controls(action)

    def run_advanced_demo_logic(self, packet: GameTickPacket):
        """
        This function is now deprecated in favor of run_advanced_demo_logic_every_tick.
        It is kept to prevent errors if it's called from somewhere else, but it does nothing.
        """
        pass

    def continue_demo_dodge(self, packet: GameTickPacket) -> np.ndarray:
        """
        Handles the multi-stage execution of any dodge.
        Simplified to remove speedflips - only basic flips are performed.
        """
        time_since_dodge_start = packet.game_info.seconds_elapsed - self.dodge_start_time
        
        # Start with the saved action plan
        action = self.dodge_action.copy()
        action[0] = 1.0  # Always throttle
        action[6] = 1.0  # Always boost during the dodge

        # Stage 1: The second jump to initiate the flip.
        if 0.05 < time_since_dodge_start < 0.15:
            action[5] = 1.0  # Second jump
        else:
            action[5] = 0.0  # Ensure we don't jump after this window

        # No flip cancel for basic flips - just let the flip complete naturally
        
        # Stage 2: The dodge is complete.
        if time_since_dodge_start >= 0.5:  # Reduced time since no cancel needed
            self.is_dodging = False
        
        return action

    def choose_demo_target(self, packet: GameTickPacket):
        """Selects the best enemy to demolish, prioritizing the closest one."""
        my_car = packet.game_cars[self.index]
        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        best_target_index = -1
        min_dist = float('inf')

        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            if i == self.index or car.team == self.team or car.is_demolished:
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

    def is_demo_target_valid(self, packet: GameTickPacket) -> bool:
        """Checks if the current target is still a valid opponent."""
        if self.demo_target_index == -1 or self.demo_target_index >= packet.num_cars:
            return False
        target = packet.game_cars[self.demo_target_index]
        return not target.is_demolished and target.team != self.team

    def predict_target_motion(self, packet: GameTickPacket):
        """
        Enhanced prediction with acceleration and advanced physics modeling.
        """
        self.demo_target_car = packet.game_cars[self.demo_target_index]
        car = self.demo_target_car
        
        self.demo_predicted_path = []
        pos = Vec3.from_rlbot_vector(car.physics.location)
        vel = Vec3.from_rlbot_vector(car.physics.velocity)
        ang_vel = Vec3.from_rlbot_vector(car.physics.angular_velocity)
        
        # Physics constants
        time_step = 1 / 120.0  # Higher resolution
        gravity = 650.0
        air_drag_coefficient = 0.0305  # RL air resistance
        ground_friction = 0.99  # Approximate
        
        # Get car orientation for thrust calculations
        car_forward = Vec3(
            math.cos(car.physics.rotation.yaw) * math.cos(car.physics.rotation.pitch),
            math.sin(car.physics.rotation.yaw) * math.cos(car.physics.rotation.pitch),
            math.sin(car.physics.rotation.pitch)
        )
        
        for i in range(180):  # 1.5 seconds at 120Hz
            # Store position
            self.demo_predicted_path.append(Vec3(pos.x, pos.y, pos.z))
            
            # Update position
            pos = pos + vel * time_step
            
            if car.has_wheel_contact:
                # Ground physics
                
                # Apply turning
                if abs(ang_vel.z) > 0.01:
                    # Calculate turning radius and adjust velocity
                    speed = vel.magnitude()
                    if speed > 100:
                        turn_radius = speed / abs(ang_vel.z)
                        centripetal_accel = speed * abs(ang_vel.z)
                        
                        # Rotate velocity vector
                        yaw_change = ang_vel.z * time_step
                        cos_yaw = math.cos(yaw_change)
                        sin_yaw = math.sin(yaw_change)
                        new_vel_x = vel.x * cos_yaw - vel.y * sin_yaw
                        new_vel_y = vel.x * sin_yaw + vel.y * cos_yaw
                        vel.x = new_vel_x
                        vel.y = new_vel_y
                
                # Apply ground friction
                vel = vel * ground_friction
                
                # Estimate acceleration/deceleration
                if car.boost > 0:
                    # Assume they might boost
                    boost_accel = 991.666
                    accel_vector = car_forward * boost_accel * time_step
                    vel = vel + accel_vector
                    # Cap at supersonic
                    if vel.magnitude() > 2300:
                        vel = vel.normalized() * 2300
                
                # Keep grounded
                pos.z = 17.01  # RL ground height
                vel.z = 0
                
            else:
                # Aerial physics
                
                # Gravity
                vel.z -= gravity * time_step
                
                # Air drag
                drag_force = air_drag_coefficient * vel.magnitude()
                if vel.magnitude() > 0:
                    vel = vel - (vel.normalized() * drag_force * time_step)
                
                # Check for ground collision
                if pos.z <= 17.01 and vel.z <= 0:
                    pos.z = 17.01
                    vel.z = 0
                    car.has_wheel_contact = True
                
                # Arena ceiling
                if pos.z >= 2044:
                    pos.z = 2044
                    vel.z = -vel.z * 0.6  # Bounce with damping
            
            # Arena walls (simplified)
            if abs(pos.x) >= 4096:
                pos.x = 4096 * (1 if pos.x > 0 else -1)
                vel.x = -vel.x * 0.6
            if abs(pos.y) >= 5120:
                pos.y = 5120 * (1 if pos.y > 0 else -1)
                vel.y = -vel.y * 0.6

    def plan_intercept(self, my_car):
        """Enhanced intercept planning with acceleration modeling."""
        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        my_vel = Vec3.from_rlbot_vector(my_car.physics.velocity)
        my_boost = my_car.boost
        
        self.demo_intercept_point = None
        best_intercept_score = float('-inf')
        
        # Physics constants for our car
        boost_accel = 991.666
        throttle_accel = 1600.0  # Approximate
        
        for i, future_pos in enumerate(self.demo_predicted_path):
            time_in_future = (i + 1) * (1 / 120.0)
            
            # Vector to intercept point
            intercept_vector = future_pos - my_pos
            intercept_distance = intercept_vector.magnitude()
            
            if intercept_distance < 1:
                continue
                
            intercept_direction = intercept_vector.normalized()
            
            # Calculate required average speed
            required_avg_speed = intercept_distance / time_in_future
            
            # Calculate if we can reach this speed
            current_speed_toward = my_vel.dot(intercept_direction)
            boost_time_available = min(my_boost / 33.3, time_in_future)
            no_boost_time = time_in_future - boost_time_available
            
            # Kinematic equation for reachable distance
            reachable_distance = (
                current_speed_toward * time_in_future +
                0.5 * boost_accel * boost_time_available**2 +
                0.5 * throttle_accel * no_boost_time**2 +
                boost_accel * boost_time_available * no_boost_time
            )
            
            # Check if we can reach it
            if reachable_distance >= intercept_distance:
                # Calculate intercept quality score
                alignment = my_vel.normalized().dot(intercept_direction) if my_vel.magnitude() > 100 else 0
                time_efficiency = 1.0 / (1.0 + time_in_future)  # Prefer sooner intercepts
                speed_score = min(2300, current_speed_toward + boost_accel * boost_time_available) / 2300
                
                score = alignment * 0.3 + time_efficiency * 0.5 + speed_score * 0.2
                
                if score > best_intercept_score:
                    best_intercept_score = score
                    self.demo_intercept_point = future_pos

    def execute_chase(self, my_car, packet: GameTickPacket) -> np.ndarray:
        """Controls steering, throttle, and boost, and decides when to dodge."""
        action = np.zeros(8)

        # Ensure we have valid target data
        if self.demo_target_car is None or self.demo_target_index < 0:
            action[0] = 1.0  # Default throttle forward
            return action

        # Basic chase logic - this section was incomplete in the original
        my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
        target_pos = Vec3.from_rlbot_vector(self.demo_target_car.physics.location)
        
        # Use intercept point if available, otherwise target current position
        chase_target = self.demo_intercept_point if self.demo_intercept_point else target_pos
        
        # Calculate direction to target
        direction = (chase_target - my_pos).normalized()
        
        # Calculate car's forward direction
        my_yaw = my_car.physics.rotation.yaw
        car_forward = Vec3(math.cos(my_yaw), math.sin(my_yaw), 0)
        
        # Calculate steering angle
        cross_product = car_forward.x * direction.y - car_forward.y * direction.x
        dot_product = car_forward.dot(direction)
        
        # Steering control
        action[1] = np.clip(cross_product * 3.0, -1.0, 1.0)  # Steer
        
        # Throttle and boost
        action[0] = 1.0  # Full throttle
        action[6] = 1.0 if my_car.boost > 0 else 0.0  # Boost if available
        
        # Simple dodge logic when close
        distance = my_pos.distance(chase_target)
        if distance < 200 and not self.is_dodging:
            # Initiate dodge
            self.is_dodging = True
            self.dodge_start_time = packet.game_info.seconds_elapsed
            self.dodge_action = action.copy()
            action[5] = 1.0  # Jump
        
        return action

    # Missing methods that are referenced but not implemented
    def update_controls(self, action):
        """Update the controls based on the action array."""
        if not hasattr(self, 'controls') or self.controls is None:
            self.controls = SimpleControllerState()
        
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0.5
        self.controls.boost = action[6] > 0.5
        self.controls.handbrake = action[7] > 0.5

    def maybe_do_kickoff(self, packet: GameTickPacket, ticks_elapsed):
        """Handle kickoff logic."""
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index == -1:
                self.kickoff_index = 0
            
            if self.kickoff_index < len(KICKOFF_NUMPY):
                self.action = KICKOFF_NUMPY[self.kickoff_index]
                self.kickoff_index += 1
        else:
            self.kickoff_index = -1

    def toxicity(self, packet: GameTickPacket):
        """Handle toxic behavior - placeholder implementation."""
        # This method was referenced but not fully implemented in the original
        pass