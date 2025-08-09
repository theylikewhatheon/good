# Self-Questioning Features in RLGym-PPO

## Overview

The RLGym-PPO library now includes "self-questioning" behavior to reduce the need for manual debugging and intervention. These features help catch common issues early and provide helpful guidance when problems occur.

## What's New

### 🤔 Configuration Validation

The Learner now asks itself questions about your configuration and warns about potential issues:

- **Batch Size Relationships**: Warns if PPO batch size is too small relative to timesteps per iteration
- **Experience Buffer Sizing**: Checks if buffer size is appropriate for iteration size
- **Learning Rate Sanity**: Warns about learning rates that might be too high or low
- **Resource Usage**: Validates process count against available CPU cores
- **GPU Memory**: Estimates memory usage and warns about potential out-of-memory issues
- **Save Intervals**: Suggests more frequent saving for long training runs

### 💾 Save/Load Safety

Enhanced checkpoint handling with user prompts:

- **Existing Checkpoint Detection**: Asks before overwriting existing checkpoints
- **Data Loss Prevention**: Prompts when save folders already contain data
- **Better Error Messages**: Provides helpful debugging tips when load fails
- **Checkpoint Validation**: Checks for incomplete or corrupted checkpoints

### 📊 Training Monitoring

Real-time performance analysis during training:

- **Performance Trends**: Detects when reward is declining and suggests causes
- **Progress Celebration**: Acknowledges significant improvements
- **Stagnation Detection**: Identifies when training has plateaued
- **Actionable Suggestions**: Provides specific recommendations for each situation

### 🛠️ Better Error Handling

Improved error messages with context:

- **Clear File Not Found**: Explains what might have gone wrong when loading fails
- **Configuration Conflicts**: Detailed explanation of parameter conflicts
- **System Resource Issues**: Guidance for hardware-related problems

## Examples

### Basic Usage (Same as Before)

```python
from rlgym_ppo import Learner

learner = Learner(
    my_env_function,
    n_proc=32,
    ppo_batch_size=50000,
    # ... other parameters
)
learner.learn()
```

### What You'll See Now

```
🤔 Asking myself some questions about your configuration...
✅ Configuration validation complete!

🤔 I found existing checkpoints in data/checkpoints/rlgym-ppo-run:
   - 100000
   - 200000
❓ Should I continue? This might overwrite existing checkpoints.
```

During training:
```
🤔 Performance seems to be declining...
   Recent average reward: 0.850
   Previous average reward: 1.200
   Questions to consider:
   - Is the learning rate too high causing instability?
   - Has the environment or reward function changed?
   - Are we overfitting to early episodes?
```

## Interactive Features

The library now asks questions in several scenarios:

1. **Startup**: Configuration validation with warnings
2. **Save Conflicts**: Prompts before overwriting data
3. **Performance Issues**: Automatic detection and suggestions
4. **Load Errors**: Helpful debugging information

## Benefits

- **Reduced Debugging Time**: Catch issues before they cause problems
- **Better Learning**: Get suggestions for improving training performance
- **Data Safety**: Prevent accidental loss of training progress
- **Educational**: Learn about RL best practices through the suggestions

## Non-Interactive Mode

All prompts gracefully handle non-interactive environments (like servers or scripts) by:
- Using sensible defaults
- Adding timestamps to prevent conflicts
- Logging decisions for review

The self-questioning features are designed to help both beginners and experts by providing contextual guidance when it's most needed.