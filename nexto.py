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

        try:
            # === NEUROSCIENCE-LEVEL DATA ACQUISITION ===
            
            # Precise timing data
            current_time = packet.game_info.seconds_elapsed
            physics_tick_rate = 120.0  # Hz
            tick_duration = 1.0 / physics_tick_rate
            
            # My car complete state vector
            my_pos = Vec3.from_rlbot_vector(my_car.physics.location)
            my_vel = Vec3.from_rlbot_vector(my_car.physics.velocity)
            my_ang_vel = Vec3.from_rlbot_vector(my_car.physics.angular_velocity)
            my_speed = my_vel.magnitude()
            
            # Rotation matrix components
            my_yaw = my_car.physics.rotation.yaw
            my_pitch = my_car.physics.rotation.pitch
            my_roll = my_car.physics.rotation.roll
            
            # Complete orientation basis vectors
            cos_yaw, sin_yaw = math.cos(my_yaw), math.sin(my_yaw)
            cos_pitch, sin_pitch = math.cos(my_pitch), math.sin(my_pitch)
            cos_roll, sin_roll = math.cos(my_roll), math.sin(my_roll)
            
            # Forward vector (car's local X-axis in world space)
            my_forward = Vec3(
                cos_yaw * cos_pitch,
                sin_yaw * cos_pitch,
                sin_pitch
            )
            
            # Right vector (car's local Y-axis in world space)
            my_right = Vec3(
                cos_yaw * sin_roll * sin_pitch + sin_yaw * cos_roll,
                sin_yaw * sin_roll * sin_pitch - cos_yaw * cos_roll,
                -sin_roll * cos_pitch
            )
            
            # Up vector (car's local Z-axis in world space)
            my_up = Vec3(
                -cos_yaw * cos_roll * sin_pitch + sin_yaw * sin_roll,
                -sin_yaw * cos_roll * sin_pitch - cos_yaw * sin_roll,
                cos_roll * cos_pitch
            )
            
            # My car state details
            my_boost = my_car.boost
            my_on_ground = my_car.has_wheel_contact
            my_jumped = my_car.jumped
            my_double_jumped = my_car.double_jumped
            
            # Wheel contact analysis
            wheels_in_contact = sum([
                my_car.physics.wheels[0].is_wheel_in_contact,
                my_car.physics.wheels[1].is_wheel_in_contact,
                my_car.physics.wheels[2].is_wheel_in_contact,
                my_car.physics.wheels[3].is_wheel_in_contact
            ])
            partial_contact = 0 < wheels_in_contact < 4
            
            # Target complete state analysis
            target_pos = Vec3.from_rlbot_vector(self.demo_target_car.physics.location)
            target_vel = Vec3.from_rlbot_vector(self.demo_target_car.physics.velocity)
            target_ang_vel = Vec3.from_rlbot_vector(self.demo_target_car.physics.angular_velocity)
            target_speed = target_vel.magnitude()
            
            # Target orientation - COMPLETE ANALYSIS
            target_yaw = self.demo_target_car.physics.rotation.yaw
            target_pitch = self.demo_target_car.physics.rotation.pitch
            target_roll = self.demo_target_car.physics.rotation.roll
            
            # Target orientation vectors
            target_cos_yaw, target_sin_yaw = math.cos(target_yaw), math.sin(target_yaw)
            target_cos_pitch, target_sin_pitch = math.cos(target_pitch), math.sin(target_pitch)
            target_cos_roll, target_sin_roll = math.cos(target_roll), math.sin(target_roll)
            
            target_forward = Vec3(
                target_cos_yaw * target_cos_pitch,
                target_sin_yaw * target_cos_pitch,
                target_sin_pitch
            )
            
            target_right = Vec3(
                target_cos_yaw * target_sin_roll * target_sin_pitch + target_sin_yaw * target_cos_roll,
                target_sin_yaw * target_sin_roll * target_sin_pitch - target_cos_yaw * target_cos_roll,
                -target_sin_roll * target_cos_pitch
            )
            
            target_up = Vec3(
                -target_cos_yaw * target_cos_roll * target_sin_pitch + target_sin_yaw * target_sin_roll,
                -target_sin_yaw * target_cos_roll * target_sin_pitch - target_cos_yaw * target_sin_roll,
                target_cos_roll * target_cos_pitch
            )
            
            # Target state details
            target_boost = self.demo_target_car.boost
            target_on_ground = self.demo_target_car.has_wheel_contact
            target_jumped = self.demo_target_car.jumped
            target_double_jumped = self.demo_target_car.double_jumped
            
            # Target wheel analysis
            target_wheels_in_contact = sum([
                self.demo_target_car.physics.wheels[0].is_wheel_in_contact,
                self.demo_target_car.physics.wheels[1].is_wheel_in_contact,
                self.demo_target_car.physics.wheels[2].is_wheel_in_contact,
                self.demo_target_car.physics.wheels[3].is_wheel_in_contact
            ])
            target_partial_contact = 0 < target_wheels_in_contact < 4
            
            # === CRITICAL: JUMP DETECTION & TRACKING ===
            
            # Track when target jumped
            if not hasattr(self, 'target_jump_tracking'):
                self.target_jump_tracking = {
                    'last_on_ground': True,
                    'jump_start_time': 0,
                    'jump_start_height': 0,
                    'jump_start_velocity': 0,
                    'time_in_air': 0,
                    'has_dodged': False,
                    'dodge_time': 0,
                    'jump_hold_duration': 0
                }
            
            # Detect new jump
            if self.target_jump_tracking['last_on_ground'] and not target_on_ground:
                self.target_jump_tracking['jump_start_time'] = current_time
                self.target_jump_tracking['jump_start_height'] = target_pos.z
                self.target_jump_tracking['jump_start_velocity'] = target_vel.z
                self.target_jump_tracking['has_dodged'] = False
                self.target_jump_tracking['time_in_air'] = 0
            
            # Update jump tracking
            if not target_on_ground:
                self.target_jump_tracking['time_in_air'] = current_time - self.target_jump_tracking['jump_start_time']
                
                # Detect if they've dodged (sudden velocity change)
                if not self.target_jump_tracking['has_dodged'] and hasattr(self, 'last_target_vel_z'):
                    vel_change = abs(target_vel.z - self.last_target_vel_z)
                    if vel_change > 200 and self.target_jump_tracking['time_in_air'] > 0.05:
                        self.target_jump_tracking['has_dodged'] = True
                        self.target_jump_tracking['dodge_time'] = current_time
            
            self.target_jump_tracking['last_on_ground'] = target_on_ground
            self.last_target_vel_z = target_vel.z
            
            # Calculate jump physics
            gravity = 650.0
            jump_impulse = 292.0
            max_jump_hold_time = 0.2
            jump_hold_boost = 1458.333  # Additional velocity from holding jump
            
            # Estimate their jump hold duration based on current height/velocity
            if not target_on_ground and self.target_jump_tracking['time_in_air'] < max_jump_hold_time:
                # Back-calculate jump hold time from current velocity
                expected_vel_no_hold = self.target_jump_tracking['jump_start_velocity'] + jump_impulse - gravity * self.target_jump_tracking['time_in_air']
                actual_vel = target_vel.z
                vel_difference = actual_vel - expected_vel_no_hold
                estimated_hold_time = vel_difference / jump_hold_boost
                self.target_jump_tracking['jump_hold_duration'] = np.clip(estimated_hold_time, 0, max_jump_hold_time)
            
            # === MICRO-DISTANCE CALCULATIONS ===
            
            # Separate component distances
            dx = target_pos.x - my_pos.x
            dy = target_pos.y - my_pos.y
            dz = target_pos.z - my_pos.z
            
            # Multiple distance metrics
            ground_distance = math.sqrt(dx**2 + dy**2)
            height_difference = abs(dz)
            true_3d_distance = math.sqrt(dx**2 + dy**2 + dz**2)
            manhattan_distance = abs(dx) + abs(dy) + abs(dz)
            
            # Surface-to-surface distance (accounting for car dimensions and orientation)
            # Hitbox dimensions (assuming Octane for now - should check car type)
            my_hitbox_length = 118.0074
            my_hitbox_width = 84.19941
            my_hitbox_height = 36.15907
            
            target_hitbox_length = 118.0074  # Should check actual car type
            target_hitbox_width = 84.19941
            target_hitbox_height = 36.15907
            
            # Calculate effective radii based on orientation
            my_radius = (my_hitbox_length / 2) * abs(my_forward.dot((target_pos - my_pos).normalized()))
            target_radius = (target_hitbox_length / 2) * abs(target_forward.dot((my_pos - target_pos).normalized()))
            surface_distance = true_3d_distance - my_radius - target_radius
            
            # === VELOCITY DECOMPOSITION ===
            
            # Direction vectors
            if ground_distance > 0.01:
                direction_to_target_ground = Vec3(dx / ground_distance, dy / ground_distance, 0.0)
                direction_to_target_3d = Vec3(dx / true_3d_distance, dy / true_3d_distance, dz / true_3d_distance)
            else:
                # Target is directly above/below us
                direction_to_target_ground = Vec3(1.0, 0.0, 0.0)  # Default forward
                direction_to_target_3d = Vec3(0.0, 0.0, 1.0 if dz > 0 else -1.0)
            
            # Velocity components relative to target direction
            my_vel_toward_target = my_vel.dot(direction_to_target_3d)
            my_vel_perpendicular = my_vel.magnitude() - abs(my_vel_toward_target)
            
            target_vel_toward_me = target_vel.dot(direction_to_target_3d * -1)
            target_vel_perpendicular = target_vel.magnitude() - abs(target_vel_toward_me)
            
            # Relative velocity analysis
            relative_velocity = target_vel - my_vel
            relative_speed = relative_velocity.magnitude()
            closing_speed = -relative_velocity.dot(direction_to_target_3d)
            
            # Angular velocity decomposition
            my_angular_velocity_toward_target = my_ang_vel.dot(direction_to_target_3d)
            target_angular_velocity_toward_me = target_ang_vel.dot(direction_to_target_3d * -1)
            
            # === INTERCEPT CALCULATIONS ===
            
            # Determine target point (use intercept if available, otherwise current position)
            if self.demo_intercept_point is not None:
                target_point = self.demo_intercept_point
                intercept_vector = target_point - my_pos
                intercept_distance = intercept_vector.magnitude()
                
                if intercept_distance > 0.01:
                    intercept_direction = intercept_vector.normalized()
                else:
                    intercept_direction = my_forward
            else:
                # Fallback to current target position
                target_point = target_pos
                intercept_direction = direction_to_target_3d
                intercept_distance = true_3d_distance
            
            # === STEERING CALCULATION ===
            
            # Calculate desired heading in world space
            desired_heading = intercept_direction
            
            # Project desired heading onto ground plane for steering
            if abs(desired_heading.z) < 0.99:  # Not purely vertical
                ground_heading = Vec3(desired_heading.x, desired_heading.y, 0.0).normalized()
            else:
                ground_heading = my_forward  # Maintain current heading if target is directly above/below
            
            # Calculate angle between current forward and desired heading
            heading_dot = my_forward.dot(ground_heading)
            heading_dot = np.clip(heading_dot, -1.0, 1.0)  # Prevent floating point errors
            heading_angle = math.acos(heading_dot)
            
            # Determine turn direction using cross product
            cross_product = Vec3(
                my_forward.y * ground_heading.z - my_forward.z * ground_heading.y,
                my_forward.z * ground_heading.x - my_forward.x * ground_heading.z,
                my_forward.x * ground_heading.y - my_forward.y * ground_heading.x
            )
            turn_direction = 1.0 if cross_product.z > 0 else -1.0
            
            # Calculate steering input with smooth ramping
            max_steer_angle = math.pi / 3  # 60 degrees max effective steering
            steer_ratio = heading_angle / max_steer_angle
            steer_ratio = np.clip(steer_ratio, 0.0, 1.0)
            
            # Apply smooth steering curve for better control
            steering_curve = math.sin(steer_ratio * math.pi / 2)  # Sine curve for smooth acceleration
            steering_input = steering_curve * turn_direction
            
            # Speed-based steering adjustment
            speed_factor = np.clip(my_speed / 1000.0, 0.2, 1.0)  # Reduce steering at low speeds
            steering_input *= speed_factor
            
            action[1] = np.clip(steering_input, -1.0, 1.0)  # Steer
            
            # === THROTTLE AND BOOST CONTROL ===
            
            # Base throttle calculation
            if intercept_distance > 50:
                # Long range: always accelerate
                throttle = 1.0
                should_boost = my_boost > 10 and my_speed < 2200
            else:
                # Short range: modulate based on approach angle and relative velocity
                approach_efficiency = max(0.1, my_vel.dot(intercept_direction))
                throttle = np.clip(approach_efficiency / max(my_speed, 100), 0.0, 1.0)
                should_boost = my_boost > 5 and closing_speed > 0 and intercept_distance > 20
            
            action[0] = throttle  # Throttle
            action[6] = 1.0 if should_boost else 0.0  # Boost
            
            # === JUMP AND DODGE LOGIC ===
            
            # Determine if we should initiate a dodge
            should_dodge = False
            dodge_direction = Vec3(0, 0, 0)
            
            if (my_on_ground and not self.is_dodging and 
                intercept_distance < 300 and intercept_distance > 80 and
                my_speed > 800 and closing_speed > 200):
                
                # Calculate optimal dodge direction
                if abs(intercept_direction.x) > abs(intercept_direction.y):
                    # More X-dominant: forward/backward dodge
                    dodge_direction = Vec3(1.0 if intercept_direction.x > 0 else -1.0, 0.0, 0.0)
                else:
                    # More Y-dominant: left/right dodge
                    dodge_direction = Vec3(0.0, 1.0 if intercept_direction.y > 0 else -1.0, 0.0)
                
                should_dodge = True
            
            if should_dodge:
                # Initiate dodge sequence
                self.is_dodging = True
                self.dodge_start_time = current_time
                
                # Set up dodge action for the sequence
                self.dodge_action = np.zeros(8)
                self.dodge_action[0] = 1.0  # Throttle
                self.dodge_action[6] = 1.0  # Boost
                
                # Set dodge direction
                if abs(dodge_direction.x) > 0.5:
                    self.dodge_action[2] = dodge_direction.x  # Pitch (forward/backward)
                if abs(dodge_direction.y) > 0.5:
                    self.dodge_action[3] = dodge_direction.y  # Yaw (left/right)
                
                # First jump
                action[5] = 1.0  # Jump
            
            # === DEFENSIVE MANEUVERS ===
            
            # If target is moving away rapidly, try to anticipate
            if closing_speed < -500 and intercept_distance < 500:
                # Target is escaping: aim further ahead
                prediction_time = intercept_distance / max(my_speed, 500)
                predicted_pos = target_pos + target_vel * prediction_time
                action[1] *= 1.5  # More aggressive steering
                action[6] = 1.0 if my_boost > 20 else 0.0  # Boost to catch up
            
            # === AERIAL CONSIDERATIONS ===
            
            if not my_on_ground and height_difference > 50:
                # We're airborne: adjust for aerial maneuvering
                action[2] = np.clip(dz / height_difference, -1.0, 1.0)  # Pitch toward target
                action[4] = np.clip(-my_ang_vel.x * 0.5, -1.0, 1.0)  # Roll correction
                
                # Air roll for better orientation
                if abs(my_roll) > 0.5:
                    action[4] = -np.sign(my_roll) * 0.8
            
            # === FINAL SAFETY CHECKS ===
            
            # Ensure all values are within valid ranges
            for i in range(len(action)):
                action[i] = np.clip(action[i], -1.0, 1.0)
            
            return action
            
        except Exception as e:
            # Fallback: if anything goes wrong, just drive forward
            print(f"Demo chase error: {e}")
            action = np.zeros(8)
            action[0] = 1.0  # Throttle
            return action

    def update_controls(self, action):
        """Convert numpy action array to SimpleControllerState."""
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0.5
        self.controls.boost = action[6] > 0.5
        self.controls.handbrake = action[7] > 0.5

    def maybe_do_kickoff(self, packet: GameTickPacket, ticks_elapsed):
        """Handle hardcoded kickoff sequence."""
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index == -1:
                self.kickoff_index = 0
        else:
            if self.kickoff_index != -1:
                self.kickoff_index = -1

        if self.kickoff_index != -1:
            if self.kickoff_index < len(KICKOFF_NUMPY):
                kickoff_action = KICKOFF_NUMPY[self.kickoff_index]
                self.update_controls(kickoff_action)
                self.kickoff_index += ticks_elapsed
            else:
                self.kickoff_index = -1

    def toxicity(self, packet: GameTickPacket):
        """Handle toxic behavior detection and response."""
        # This is a placeholder for toxicity handling logic
        # Implementation would depend on specific toxicity detection requirements
        pass