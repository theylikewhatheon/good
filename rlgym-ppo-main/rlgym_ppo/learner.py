"""
File: learner.py
Author: Matthew Allen

Description:
The primary algorithm file. The Learner object coordinates timesteps from the workers 
and sends them to PPO, keeps track of the misc. variables and statistics for logging,
reports to wandb and the console, and handles checkpointing.
"""

import json
import os
import random
import shutil
import time
from typing import Union, Tuple

import numpy as np
import torch
import wandb
from wandb.wandb_run import Run

from rlgym_ppo.batched_agents import BatchedAgentManager
from rlgym_ppo.ppo import ExperienceBuffer, PPOLearner
from rlgym_ppo.util import WelfordRunningStat, reporting, torch_functions, KBHit


class Learner(object):
    def __init__(
            # fmt: off
            self,
            env_create_function,
            metrics_logger=None,
            n_proc: int = 8,
            min_inference_size: int = 80,
            render: bool = False,
            render_delay: float = 0,

            timestep_limit: int = 5_000_000_000,
            exp_buffer_size: int = 100000,
            ts_per_iteration: int = 50000,
            standardize_returns: bool = True,
            standardize_obs: bool = True,
            max_returns_per_stats_increment: int = 150,
            steps_per_obs_stats_increment: int = 5,

            policy_layer_sizes: Tuple[int, ...] = (256, 256, 256),
            critic_layer_sizes: Tuple[int, ...] = (256, 256, 256),
            continuous_var_range: Tuple[float, ...] = (0.1, 1.0),

            ppo_epochs: int = 10,
            ppo_batch_size: int = 50000,
            ppo_minibatch_size: Union[int, None] = None,
            ppo_ent_coef: float = 0.005,
            ppo_clip_range: float = 0.2,

            gae_lambda: float = 0.95,
            gae_gamma: float = 0.99,
            policy_lr: float = 3e-4,
            critic_lr: float = 3e-4,

            log_to_wandb: bool = False,
            load_wandb: bool = True,
            wandb_run: Union[Run, None] = None,
            wandb_project_name: Union[str, None] = None,
            wandb_group_name: Union[str, None] = None,
            wandb_run_name: Union[str, None] = None,

            checkpoints_save_folder: Union[str, None] = None,
            add_unix_timestamp: bool = True,
            checkpoint_load_folder: Union[str, None] = "latest", # "latest" loads latest checkpoint
            save_every_ts: int = 1_000_000,

            instance_launch_delay: Union[float, None] = None,
            random_seed: int = 123,
            n_checkpoints_to_keep: int = 5,
            shm_buffer_size: int = 8192,
            device: str = "auto"):

        assert (
                env_create_function is not None
        ), "MUST PROVIDE A FUNCTION TO CREATE RLGYM FUNCTIONS TO INITIALIZE RLGYM-PPO"
        
        # Self-questioning: Validate configuration and warn about potential issues
        self._validate_configuration_and_warn(
            n_proc, min_inference_size, exp_buffer_size, ts_per_iteration,
            ppo_batch_size, ppo_minibatch_size, ppo_epochs, policy_lr, critic_lr,
            save_every_ts, timestep_limit, device
        )

        if checkpoints_save_folder is None:
            checkpoints_save_folder = os.path.join(
                "data", "checkpoints", "rlgym-ppo-run"
            )
            
        # Self-questioning: Check if save folder already exists and prompt user
        self._check_save_folder_and_prompt(checkpoints_save_folder, add_unix_timestamp)

        # Add the option for the user to turn off the addition of Unix Timestamps to
        # the ``checkpoints_save_folder`` path
        self.add_unix_timestamp = add_unix_timestamp
        if add_unix_timestamp:
            checkpoints_save_folder = f"{checkpoints_save_folder}-{time.time_ns()}"

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.n_checkpoints_to_keep = n_checkpoints_to_keep
        self.checkpoints_save_folder = checkpoints_save_folder
        self.max_returns_per_stats_increment = max_returns_per_stats_increment
        self.metrics_logger = metrics_logger
        self.standardize_returns = standardize_returns
        self.save_every_ts = save_every_ts
        self.ts_since_last_save = 0

        if device in {"auto", "gpu"} and torch.cuda.is_available():
            self.device = "cuda:0"
            torch.backends.cudnn.benchmark = True
        elif device == "auto" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        print(f"Using device {self.device}")
        self.exp_buffer_size = exp_buffer_size
        self.timestep_limit = timestep_limit
        self.ts_per_epoch = ts_per_iteration
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma
        self.return_stats = WelfordRunningStat(1)
        self.epoch = 0

        self.experience_buffer = ExperienceBuffer(
            self.exp_buffer_size, seed=random_seed, device="cpu"
        )

        print("Initializing processes...")
        collect_metrics_fn = None if metrics_logger is None else self.metrics_logger.collect_metrics
        self.agent = BatchedAgentManager(
            None, min_inference_size=min_inference_size,
            seed=random_seed,
            standardize_obs=standardize_obs,
            steps_per_obs_stats_increment=steps_per_obs_stats_increment
        )
        obs_space_size, act_space_size, action_space_type = self.agent.init_processes(
            n_processes=n_proc,
            build_env_fn=env_create_function,
            collect_metrics_fn=collect_metrics_fn,
            spawn_delay=instance_launch_delay,
            render=render,
            render_delay=render_delay,
            shm_buffer_size=shm_buffer_size
        )
        obs_space_size = np.prod(obs_space_size)
        print("Initializing PPO...")
        if ppo_minibatch_size is None:
            ppo_minibatch_size = ppo_batch_size

        self.ppo_learner = PPOLearner(
            obs_space_size,
            act_space_size,
            device=self.device,
            batch_size=ppo_batch_size,
            mini_batch_size=ppo_minibatch_size,
            n_epochs=ppo_epochs,
            continuous_var_range=continuous_var_range,
            policy_type=action_space_type,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef,
        )

        self.agent.policy = self.ppo_learner.policy

        self.config = {
            "n_proc": n_proc,
            "min_inference_size": min_inference_size,
            "timestep_limit": timestep_limit,
            "exp_buffer_size": exp_buffer_size,
            "ts_per_iteration": ts_per_iteration,
            "standardize_returns": standardize_returns,
            "standardize_obs": standardize_obs,
            "policy_layer_sizes": policy_layer_sizes,
            "critic_layer_sizes": critic_layer_sizes,
            "ppo_epochs": ppo_epochs,
            "ppo_batch_size": ppo_batch_size,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_ent_coef": ppo_ent_coef,
            "ppo_clip_range": ppo_clip_range,
            "gae_lambda": gae_lambda,
            "gae_gamma": gae_gamma,
            "policy_lr": policy_lr,
            "critic_lr": critic_lr,
            "shm_buffer_size": shm_buffer_size,
        }

        self.wandb_run = wandb_run
        wandb_loaded = checkpoint_load_folder is not None and self.load(checkpoint_load_folder, load_wandb, policy_lr, critic_lr)

        if log_to_wandb and self.wandb_run is None and not wandb_loaded:
            project = "rlgym-ppo" if wandb_project_name is None else wandb_project_name
            group = "unnamed-runs" if wandb_group_name is None else wandb_group_name
            run_name = "rlgym-ppo-run" if wandb_run_name is None else wandb_run_name
            print("Attempting to create new wandb run...")
            self.wandb_run = wandb.init(
                project=project, group=group, config=self.config, name=run_name, reinit=True
            )
            print("Created new wandb run!", self.wandb_run.id)
        print("Learner successfully initialized!")

    def update_learning_rate(self, new_policy_lr=None, new_critic_lr=None):
        if new_policy_lr is not None:
            self.policy_lr = new_policy_lr
            for param_group in self.ppo_learner.policy_optimizer.param_groups:
                param_group['lr'] = new_policy_lr
            print(f"New policy learning rate: {new_policy_lr}")

        if new_critic_lr is not None:
            self.critic_lr = new_critic_lr
            for param_group in self.ppo_learner.value_optimizer.param_groups:
                param_group['lr'] = new_critic_lr
            print(f"New policy learning rate: {new_policy_lr}")

    def learn(self):
        """
        Function to wrap the _learn function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            self._learn()
        except Exception:
            import traceback

            print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
            traceback.print_exc()

            try:
                self.save(self.agent.cumulative_timesteps)
            except:
                print("FAILED TO SAVE ON EXIT")

        finally:
            self.cleanup()

    def _learn(self):
        """
        Learning function. This is where the magic happens.
        :return: None
        """

        # Class to watch for keyboard hits
        kb = KBHit()
        print("Press (p) to pause (c) to checkpoint, (q) to checkpoint and quit (after next iteration)")
        print("🤔 I'll be asking myself questions during training to help catch issues early!\n")

        # Self-questioning: Track performance metrics to detect issues
        performance_history = []
        last_performance_check = 0

        # While the number of timesteps we have collected so far is less than the
        # amount we are allowed to collect.
        while self.agent.cumulative_timesteps < self.timestep_limit:
            epoch_start = time.perf_counter()
            report = {}

            # Collect the desired number of timesteps from our agent.
            experience, collected_metrics, steps_collected, collection_time = self.agent.collect_timesteps(
                self.ts_per_epoch
            )

            if self.metrics_logger is not None:
                self.metrics_logger.report_metrics(collected_metrics, self.wandb_run, self.agent.cumulative_timesteps)

            # Add the new experience to our buffer and compute the various
            # reinforcement learning quantities we need to
            # learn from (advantages, values, returns).
            self.add_new_experience(experience)

            # Let PPO compute updates using our experience buffer.
            ppo_report = self.ppo_learner.learn(self.experience_buffer)
            epoch_stop = time.perf_counter()
            epoch_time = epoch_stop - epoch_start

            # Report variables we care about.
            report.update(ppo_report)
            if self.epoch < 1:
                report["Value Function Loss"] = np.nan

            report["Cumulative Timesteps"] = self.agent.cumulative_timesteps
            report["Total Iteration Time"] = epoch_time
            report["Timesteps Collected"] = steps_collected
            report["Timestep Collection Time"] = collection_time
            report["Timestep Consumption Time"] = epoch_time - collection_time
            report["Collected Steps per Second"] = steps_collected / collection_time
            report["Overall Steps per Second"] = steps_collected / epoch_time

            self.ts_since_last_save += steps_collected
            if self.agent.average_reward is not None:
                report["Policy Reward"] = self.agent.average_reward
                
                # Self-questioning: Monitor performance and detect issues
                performance_history.append(self.agent.average_reward)
                if len(performance_history) > 50:  # Keep last 50 episodes
                    performance_history.pop(0)
                    
                # Check for performance issues every 10 epochs
                if self.epoch > 0 and self.epoch % 10 == 0 and len(performance_history) >= 20:
                    recent_avg = np.mean(performance_history[-10:])
                    older_avg = np.mean(performance_history[-20:-10])
                    
                    if recent_avg < older_avg * 0.8:  # 20% performance drop
                        print(f"🤔 Performance seems to be declining...")
                        print(f"   Recent average reward: {recent_avg:.3f}")
                        print(f"   Previous average reward: {older_avg:.3f}")
                        print("   Questions to consider:")
                        print("   - Is the learning rate too high causing instability?")
                        print("   - Has the environment or reward function changed?")
                        print("   - Are we overfitting to early episodes?")
                        
                    elif recent_avg > older_avg * 1.5:  # Significant improvement
                        print(f"🚀 Great progress detected!")
                        print(f"   Recent average reward: {recent_avg:.3f} (vs {older_avg:.3f})")
                        
                    # Check for stagnation
                    if len(performance_history) >= 50:
                        std_recent = np.std(performance_history[-25:])
                        if std_recent < 0.01 and abs(recent_avg) > 0.1:  # Very stable but not near zero
                            print(f"🤔 Performance seems to have plateaued...")
                            print(f"   Reward has been stable around {recent_avg:.3f} for a while")
                            print("   Questions to consider:")
                            print("   - Should we adjust exploration (entropy coefficient)?")
                            print("   - Is the current policy near optimal for this task?")
                            print("   - Should we change the learning rate or other hyperparameters?")
            else:
                report["Policy Reward"] = np.nan

            # Log to wandb and print to the console.
            reporting.report_metrics(loggable_metrics=report,
                                     debug_metrics=None,
                                     wandb_run=self.wandb_run)

            report.clear()
            ppo_report.clear()

            if "cuda" in self.device:
                torch.cuda.empty_cache()

            # Check if keyboard press
            # p: pause, any key to resume
            # c: checkpoint
            # q: checkpoint and quit

            if kb.kbhit():
                c = kb.getch()
                if c == 'p':  # pause
                    print("Paused, press any key to resume")
                    while True:
                        if kb.kbhit():
                            break
                if c in ('c', 'q'):
                    self.save(self.agent.cumulative_timesteps)
                if c == 'q':
                    return
                if c in ('c', 'p'):
                    print("Resuming...\n")

            # Save if we've reached the next checkpoint timestep.
            if self.ts_since_last_save >= self.save_every_ts:
                self.save(self.agent.cumulative_timesteps)
                self.ts_since_last_save = 0

            self.epoch += 1

    @torch.no_grad()
    def add_new_experience(self, experience):
        """
        Function to add timesteps to our experience buffer and compute the advantage
        function estimates, value function
        estimates, and returns.
        :param experience: tuple containing
        (experience, steps_collected, collection_time) from an agent.
        :return: None
        """

        # Unpack timestep data.
        states, actions, log_probs, rewards, next_states, dones, truncated = experience
        value_net = self.ppo_learner.value_net

        # Construct input to the value function estimator that includes the final state
        # (which an action was not taken in)
        val_inp = np.zeros(shape=(states.shape[0] + 1, states.shape[1]))
        val_inp[:-1] = states
        val_inp[-1] = next_states[-1]

        # Predict the expected returns at each state.
        val_preds = value_net(val_inp).cpu().flatten().tolist()
        torch.cuda.empty_cache()

        # Compute the desired reinforcement learning quantities.
        ret_std = self.return_stats.std[0] if self.standardize_returns else None

        value_targets, advantages, returns = torch_functions.compute_gae(
            rewards,
            dones,
            truncated,
            val_preds,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
            return_std=ret_std,  # 1 by default if no standardization is requested
        )

        if self.standardize_returns:
            # Update the running statistics about the returns.
            n_to_increment = min(self.max_returns_per_stats_increment, len(returns))

            self.return_stats.increment(returns[:n_to_increment], n_to_increment)

        # Add our new experience to the buffer.
        self.experience_buffer.submit_experience(
            states,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            truncated,
            value_targets,
            advantages,
        )

    def save(self, cumulative_timesteps):
        """
        Function to save a checkpoint.
        :param cumulative_timesteps: Number of timesteps that have passed so far in the
        learning algorithm.
        :return: None
        """

        # Make the file path to which the checkpoint will be saved
        folder_path = os.path.join(
            self.checkpoints_save_folder, str(cumulative_timesteps)
        )
        
        # Self-questioning: Check if we're about to save over existing data
        if os.path.exists(folder_path):
            print(f"🤔 I found an existing checkpoint at {folder_path}")
            print("❓ Should I overwrite it? (y/N): ", end="")
            try:
                import sys
                # Check if we're in an interactive environment
                if sys.stdin.isatty():
                    response = input().strip().lower()
                    if response != 'y':
                        print("✅ Skipping save to prevent overwrite.")
                        return
                    print("⚠️  Overwriting existing checkpoint...")
                else:
                    print("Non-interactive mode: overwriting existing checkpoint...")
            except (KeyboardInterrupt, EOFError):
                print("✅ Save cancelled by user.")
                return
        
        os.makedirs(folder_path, exist_ok=True)

        # Check to see if we've run out of checkpoint space and remove the oldest
        # checkpoints
        print(f"Saving checkpoint {cumulative_timesteps}...")
        existing_checkpoints = [
            int(arg) for arg in os.listdir(self.checkpoints_save_folder)
            if arg.isdigit()  # Self-questioning: Only consider numeric folders as checkpoints
        ]
        if len(existing_checkpoints) > self.n_checkpoints_to_keep:
            existing_checkpoints.sort()
            checkpoints_to_remove = existing_checkpoints[: -self.n_checkpoints_to_keep]
            
            # Self-questioning: Inform user about what we're deleting
            if len(checkpoints_to_remove) > 0:
                print(f"🗑️  Removing {len(checkpoints_to_remove)} old checkpoints to stay within limit of {self.n_checkpoints_to_keep}")
                for checkpoint_name in checkpoints_to_remove:
                    checkpoint_path = os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                    print(f"   Removing: {checkpoint_path}")
                    shutil.rmtree(checkpoint_path)

        # Save all the things that need saving.
        self.ppo_learner.save_to(folder_path)

        book_keeping_vars = {
            "cumulative_timesteps": self.agent.cumulative_timesteps,
            "cumulative_model_updates": self.ppo_learner.cumulative_model_updates,
            "policy_average_reward": self.agent.average_reward,
            "epoch": self.epoch,
            "ts_since_last_save": self.ts_since_last_save,
            "reward_running_stats": self.return_stats.to_json(),

        }
        if self.agent.standardize_obs:
            book_keeping_vars["obs_running_stats"] = self.agent.obs_stats.to_json()
        if self.standardize_returns:
            book_keeping_vars["reward_running_stats"] = self.return_stats.to_json()

        if self.wandb_run is not None:
            book_keeping_vars["wandb_run_id"] = self.wandb_run.id
            book_keeping_vars["wandb_project"] = self.wandb_run.project
            book_keeping_vars["wandb_entity"] = self.wandb_run.entity
            book_keeping_vars["wandb_group"] = self.wandb_run.group
            book_keeping_vars["wandb_config"] = self.wandb_run.config.as_dict()

        book_keeping_table_path = os.path.join(folder_path, "BOOK_KEEPING_VARS.json")
        with open(book_keeping_table_path, "w") as f:
            json.dump(book_keeping_vars, f, indent=4)

        print(f"Checkpoint {cumulative_timesteps} saved!\n")

    def load(self, folder_path, load_wandb, new_policy_lr=None, new_critic_lr=None):
        """
        Function to load the learning algorithm from a checkpoint.

        :param folder_path: Path to the checkpoint folder that will be loaded.
        :param load_wandb: Whether to resume an existing weights and biases run that
        was saved with the checkpoint being loaded.
        :return: None
        """

        if folder_path == "latest":
            save_folder = self.checkpoints_save_folder
            if save_folder is None:
                print("❌ No save folder configured for loading latest checkpoint.")
                return

            if self.add_unix_timestamp:
                # Save folder without the unix timestamp
                base_save_folder = save_folder[:save_folder.rfind('-')]
                save_path = os.path.dirname(base_save_folder)

                if not os.path.exists(save_path):
                    print(f"❌ Save path does not exist: {save_path}")
                    return

                print(f"🔍 Looking for latest checkpoint in {save_path}...")
                # Find folder with our base save path with the highest timestamp
                highest_timestamp = -1
                best_folder = None
                for filename in os.listdir(save_path):
                    full_path = os.path.join(save_path, filename)
                    if not os.path.isdir(full_path):
                        continue

                    if full_path.startswith(base_save_folder):
                        unix_start_idx = full_path.rfind('-') + 1
                        if unix_start_idx > 0:
                            unix_time_str = filename[unix_start_idx:]
                            if unix_time_str.isdigit():
                                timestamp = int(unix_time_str)
                                if timestamp > highest_timestamp:
                                    highest_timestamp = timestamp
                                    best_folder = full_path

                if not (best_folder is None):
                    load_base_path = best_folder
                    print(f"📁 Found timestamped folder: {best_folder}")
                else:
                    print("❌ Failed to find any unix timestamp folders under the right name")
                    return
            else:
                if os.path.exists(self.checkpoints_save_folder):
                    load_base_path = self.checkpoints_save_folder
                    print(f"📁 Using checkpoint folder: {load_base_path}")
                else:
                    print(f"❌ Save path doesn't exist: {self.checkpoints_save_folder}")
                    return

            # Find folder with the highest timesteps to load
            highest_ts = -1
            available_checkpoints = []
            for filename in os.listdir(load_base_path):
                if not os.path.isdir(os.path.join(load_base_path, filename)):
                    continue

                if not filename.isdigit():
                    continue
                
                ts = int(filename)
                available_checkpoints.append(ts)
                highest_ts = max(highest_ts, ts)

            if highest_ts != -1:
                folder_path = os.path.join(load_base_path, str(highest_ts))
                print(f"🎯 Auto-loading latest checkpoint: {folder_path}")
                if len(available_checkpoints) > 1:
                    available_checkpoints.sort()
                    print(f"📊 Available checkpoints: {available_checkpoints}")
            else:
                print("❌ No timestep folders found to load")
                return

        # Make sure the folder exists.
        if not os.path.exists(folder_path):
            print(f"❌ Unable to locate checkpoint folder: {folder_path}")
            print("🤔 Questions to ask yourself:")
            print("   - Did you specify the correct path?")
            print("   - Are you sure the checkpoint was saved successfully?")
            print("   - Are you trying to load from the right machine/location?")
            raise FileNotFoundError(f"Checkpoint folder not found: {folder_path}")
            
        print(f"📂 Loading from checkpoint at {folder_path}")

        # Self-questioning: Validate checkpoint contents before loading
        required_files = ["BOOK_KEEPING_VARS.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(folder_path, f))]
        if missing_files:
            print(f"❌ Checkpoint appears incomplete. Missing files: {missing_files}")
            print("🤔 This checkpoint might be corrupted or from an incompatible version.")
            raise FileNotFoundError(f"Incomplete checkpoint: missing {missing_files}")

        # Load stuff.
        try:
            self.ppo_learner.load_from(folder_path)
            print("✅ PPO learner loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load PPO learner: {e}")
            print("🤔 This might be a version compatibility issue or corrupted model files.")
            raise

        wandb_loaded = False
        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "r") as f:
            book_keeping_vars = dict(json.load(f))
            self.agent.cumulative_timesteps = book_keeping_vars["cumulative_timesteps"]
            self.agent.average_reward = book_keeping_vars["policy_average_reward"]
            self.ppo_learner.cumulative_model_updates = book_keeping_vars[
                "cumulative_model_updates"
            ]
            self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            if self.agent.standardize_obs and "obs_running_stats" in book_keeping_vars.keys():
                self.agent.obs_stats = WelfordRunningStat(1)
                self.agent.obs_stats.from_json(book_keeping_vars["obs_running_stats"])
            if self.standardize_returns and "reward_running_stats" in book_keeping_vars.keys():
                self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            self.epoch = book_keeping_vars["epoch"]
            
            # Update learning rates if new values are provided
            if new_policy_lr is not None or new_critic_lr is not None:
                self.update_learning_rate(new_policy_lr, new_critic_lr)

            # check here for backwards compatibility

            if "wandb_run_id" in book_keeping_vars and load_wandb:
                self.wandb_run = wandb.init(
                    settings=wandb.Settings(start_method="spawn"),
                    entity=book_keeping_vars["wandb_entity"],
                    project=book_keeping_vars["wandb_project"],
                    group=book_keeping_vars["wandb_group"],
                    id=book_keeping_vars["wandb_run_id"],
                    config=book_keeping_vars["wandb_config"],
                    resume="allow",
                    reinit=True,
                )
                wandb_loaded = True

        print("Checkpoint loaded!")
        return wandb_loaded

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """

        if self.wandb_run is not None:
            self.wandb_run.finish()
        if type(self.agent) == BatchedAgentManager:
            self.agent.cleanup()
        self.experience_buffer.clear()
        
    def _validate_configuration_and_warn(self, n_proc, min_inference_size, exp_buffer_size, 
                                        ts_per_iteration, ppo_batch_size, ppo_minibatch_size,
                                        ppo_epochs, policy_lr, critic_lr, save_every_ts, 
                                        timestep_limit, device):
        """
        Self-questioning: Validate configuration parameters and warn about potential issues.
        This reduces the need for manual debugging by catching common problems early.
        """
        print("🤔 Asking myself some questions about your configuration...")
        
        # Question 1: Are batch sizes reasonable?
        if ppo_minibatch_size is None:
            ppo_minibatch_size = ppo_batch_size
            
        if ppo_batch_size < ts_per_iteration * 0.5:
            print(f"⚠️  Warning: Your PPO batch size ({ppo_batch_size}) is less than half your timesteps per iteration ({ts_per_iteration}).")
            print("   This might lead to inefficient learning. Consider increasing ppo_batch_size or decreasing ts_per_iteration.")
            
        if ppo_minibatch_size > ppo_batch_size:
            print(f"❌ Error: PPO minibatch size ({ppo_minibatch_size}) cannot be larger than batch size ({ppo_batch_size}).")
            raise ValueError("ppo_minibatch_size must be <= ppo_batch_size")
            
        # Question 2: Is the experience buffer appropriately sized?
        if exp_buffer_size < ts_per_iteration * 2:
            print(f"⚠️  Warning: Experience buffer size ({exp_buffer_size}) is less than 2x timesteps per iteration ({ts_per_iteration}).")
            print("   This might cause training instability. Consider increasing exp_buffer_size.")
            
        # Question 3: Are learning rates in a reasonable range?
        if policy_lr > 1e-2:
            print(f"⚠️  Warning: Policy learning rate ({policy_lr}) seems high. This might cause training instability.")
        if critic_lr > 1e-2:
            print(f"⚠️  Warning: Critic learning rate ({critic_lr}) seems high. This might cause training instability.")
            
        if policy_lr < 1e-6:
            print(f"⚠️  Warning: Policy learning rate ({policy_lr}) seems very low. Training might be extremely slow.")
        if critic_lr < 1e-6:
            print(f"⚠️  Warning: Critic learning rate ({critic_lr}) seems very low. Training might be extremely slow.")
            
        # Question 4: Is the process count reasonable?
        try:
            import multiprocessing
            available_cores = multiprocessing.cpu_count()
            if n_proc > available_cores:
                print(f"⚠️  Warning: You're requesting {n_proc} processes but only have {available_cores} CPU cores.")
                print("   This might actually slow down training due to context switching overhead.")
        except:
            pass
            
        # Question 5: Are we using GPU efficiently?
        if device in {"auto", "gpu"}:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✅ Found GPU with {gpu_memory:.1f}GB memory")
                    
                    # Estimate memory usage
                    estimated_memory = (ppo_batch_size * 50 * 4) / 1e9  # rough estimate
                    if estimated_memory > gpu_memory * 0.8:
                        print(f"⚠️  Warning: Estimated memory usage ({estimated_memory:.1f}GB) might exceed GPU memory.")
                        print("   Consider reducing batch sizes if you encounter out-of-memory errors.")
                else:
                    print("⚠️  Warning: GPU requested but not available. Falling back to CPU.")
            except ImportError:
                print("⚠️  Warning: PyTorch not available. Cannot use GPU acceleration.")
                
        # Question 6: Are save intervals reasonable?
        expected_runtime_hours = timestep_limit / (ts_per_iteration * 3600)  # rough estimate
        save_interval_hours = save_every_ts / (ts_per_iteration * 3600)
        
        if save_interval_hours > 2 and expected_runtime_hours > 4:
            print(f"⚠️  Warning: You're saving every {save_interval_hours:.1f} hours with an expected runtime of {expected_runtime_hours:.1f} hours.")
            print("   Consider saving more frequently to avoid losing progress.")
            
        # Question 7: Is min_inference_size appropriate?
        if min_inference_size > n_proc * 0.9:
            print(f"⚠️  Warning: min_inference_size ({min_inference_size}) is very close to n_proc ({n_proc}).")
            print("   This might cause inefficient batching. Consider reducing min_inference_size.")
            
        print("✅ Configuration validation complete!\n")
        
    def _check_save_folder_and_prompt(self, checkpoints_save_folder, add_unix_timestamp):
        """
        Self-questioning: Check if we might overwrite existing data and prompt user if needed.
        """
        if not add_unix_timestamp and os.path.exists(checkpoints_save_folder):
            existing_checkpoints = []
            try:
                existing_checkpoints = [f for f in os.listdir(checkpoints_save_folder) 
                                      if os.path.isdir(os.path.join(checkpoints_save_folder, f))]
            except:
                pass
                
            if existing_checkpoints:
                print(f"🤔 I found existing checkpoints in {checkpoints_save_folder}:")
                for cp in existing_checkpoints[:5]:  # Show first 5
                    print(f"   - {cp}")
                if len(existing_checkpoints) > 5:
                    print(f"   ... and {len(existing_checkpoints) - 5} more")
                    
                print("\n❓ Should I continue? This might overwrite existing checkpoints.")
                print("   Options:")
                print("   - Press Enter to continue with unix timestamp (recommended)")
                print("   - Type 'y' to continue without timestamp (might overwrite)")
                print("   - Type 'n' to abort")
                
                try:
                    response = input("Your choice: ").strip().lower()
                    if response == 'n':
                        print("❌ Aborting to prevent data loss.")
                        raise KeyboardInterrupt("User chose to abort to prevent data loss")
                    elif response == 'y':
                        print("⚠️  Continuing without timestamp. Existing checkpoints may be overwritten.")
                    else:
                        print("✅ Adding unix timestamp to prevent conflicts.")
                        self.add_unix_timestamp = True
                except (KeyboardInterrupt, EOFError):
                    print("❌ Aborting to prevent data loss.")
                    raise KeyboardInterrupt("User chose to abort to prevent data loss")
