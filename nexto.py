import numpy as np
import torch
import keyboard
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.quick_chats import QuickChats
from rlgym_compat import GameState

from agent import Agent
from nexto_obs import NextoObsBuilder, BOOST_LOCATIONS
from bot_enums import BotMode
from demo_controller import DemoController
from kickoff_controller import KickoffController


class Nexto(BaseAgent):
    """
    Advanced Rocket League bot with normal play and demo modes.
    
    Features:
    - Neural network-based decision making
    - Manual demo mode toggle with 'C' key
    - Hardcoded kickoff sequences
    - Attention weight visualization
    """
    
    # Class-level flag to prevent multiple spams
    _has_printed_init = False

    def __init__(self, name, team, index,
                 beta=1, render=False, hardcoded_kickoffs=True, stochastic_kickoffs=True):
        super().__init__(name, team, index)

        # Core components
        self.obs_builder = None
        self.agent = Agent()
        self.tick_skip = 8

        # Configuration
        self.beta = beta  # Controls randomness: 1=best, 0.5=sampling, 0=random, -1=worst
        self.render = render
        self.hardcoded_kickoffs = hardcoded_kickoffs
        self.stochastic_kickoffs = stochastic_kickoffs

        # Game state
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.field_info = None

        # Mode management
        self.mode = BotMode.NORMAL
        self.c_key_pressed_last_tick = False

        # Controller components
        self.demo_controller = DemoController()
        self.kickoff_controller = KickoffController()

        # Toxicity handling (simplified)
        self.isToxic = False
        self.toxic_stats = {
            'orange_goals': 0,
            'blue_goals': 0,
            'demo_count': 0,
            'demoed_count': 0
        }

        # Initialize only once
        if not Nexto._has_printed_init:
            print('Nexto Ready - Index:', index, 'Beta:', str(beta))
            print("Remember to run Nexto at 120fps with vsync off! "
                  "Stable 240/360 is second best if that's better for your eyes")
            print("Also check out the RLGym Twitch stream to watch live bot training and occasional showmatches!")
            Nexto._has_printed_init = True

    def initialize_agent(self, field_info):
        """Initialize the rlgym GameState object once game info is available."""
        self.field_info = field_info
        self.obs_builder = NextoObsBuilder(field_info=self.field_info)
        self.game_state = GameState(self.field_info)
        self.ticks = self.tick_skip  # Take action on first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

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
        """Main bot logic - handles mode switching and delegates to appropriate controllers."""
        # Handle mode toggle
        self._handle_mode_toggle()
        
        # Always update game state
        self._update_game_state(packet)

        # Route to appropriate logic based on mode
        if self.mode == BotMode.DEMO:
            # Demo mode runs every tick for maximum responsiveness
            action = self.demo_controller.update(packet, self.index)
            self.update_controls(action)
        else:
            # Normal mode respects tick_skip for efficiency
            self._run_normal_mode(packet)

        return self.controls

    def _handle_mode_toggle(self):
        """Handle manual mode switching with 'C' key."""
        c_key_is_down = keyboard.is_pressed('c')
        if c_key_is_down and not self.c_key_pressed_last_tick:
            if self.mode == BotMode.NORMAL:
                self.mode = BotMode.DEMO
                self.send_quick_chat(QuickChats.CHAT_TEAM_ONLY, QuickChats.Custom_Useful_Bumping)
            else:
                self.mode = BotMode.NORMAL
                self.send_quick_chat(QuickChats.CHAT_TEAM_ONLY, QuickChats.Information_Defending)
        self.c_key_pressed_last_tick = c_key_is_down

    def _update_game_state(self, packet: GameTickPacket):
        """Update game state - critical for both modes to have fresh data."""
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

    def _run_normal_mode(self, packet: GameTickPacket):
        """Run the normal Nexto brain logic with tick skipping."""
        if self.isToxic:
            self.toxicity(packet)

        self.run_nexto_logic(packet)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        if self.hardcoded_kickoffs:
            self.action = self.kickoff_controller.update(packet, self.action)

    def run_nexto_logic(self, packet: GameTickPacket):
        """The standard Nexto neural network brain logic."""
        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)

            beta = self.beta
            # Add randomness if match ended
            if packet.game_info.is_match_ended:
                beta = 0
            # Add randomness on kickoffs if enabled
            if self.stochastic_kickoffs and packet.game_info.is_kickoff_pause:
                beta = 0.5

            self.action, weights = self.agent.act(obs, beta)

            if self.render:
                positions = np.asarray([p.car_data.position for p in self.game_state.players] +
                                       [self.game_state.ball.position] +
                                       list(BOOST_LOCATIONS))
                self.render_attention_weights(weights, positions)

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

    def toxicity(self, packet: GameTickPacket):
        """Handle toxic behavior - placeholder implementation."""
        # Simplified toxicity handling
        pass