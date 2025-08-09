# Nexto Bot - Refactored Structure

A well-structured Rocket League bot built with modularity and maintainability in mind.

## Structure Overview

The original monolithic `nexto.py` file has been refactored into a modular architecture:

```
nexto/
├── __init__.py                 # Package initialization
├── nexto.py                   # Main bot class (streamlined)
├── vector_utils.py            # Vec3 vector mathematics
├── bot_enums.py               # Bot modes and states
├── physics_utils.py           # Physics calculations and predictions
├── demo_controller.py         # Demolition logic
├── kickoff_controller.py      # Kickoff sequences
├── agent.py                   # Neural network agent (placeholder)
├── nexto_obs.py              # Observation builder (placeholder)
└── test_structure.py         # Structure validation tests
```

## Key Improvements

### 1. **Separation of Concerns**
- **Vector math** isolated in `vector_utils.py`
- **Physics calculations** in `physics_utils.py`
- **Demo logic** in `demo_controller.py`
- **Kickoff handling** in `kickoff_controller.py`

### 2. **Reduced Complexity**
- Main `Nexto` class reduced from ~580 lines to ~220 lines
- Complex physics calculations moved to dedicated modules
- State management simplified with clear delegation

### 3. **Better Maintainability**
- Each module has a single responsibility
- Clear interfaces between components
- Easier to test individual components
- Reduced coupling between different features

### 4. **Improved Readability**
- Well-documented classes and methods
- Clear naming conventions
- Logical organization of related functionality

## Architecture Benefits

### Before (Monolithic)
- 580+ lines in single file
- Mixed concerns (physics, AI, demo logic, etc.)
- Hard to test individual components
- Difficult to modify without affecting other features

### After (Modular)
- Main class: ~220 lines
- Separated modules: ~50-200 lines each
- Clear boundaries between features
- Easy to test and modify individual components

## Usage

```python
from nexto import Nexto

# Create bot instance
bot = Nexto(name="Nexto", team=0, index=0, beta=1.0, render=False)

# Initialize with field info
bot.initialize_agent(field_info)

# Main game loop
while True:
    controls = bot.get_output(packet)
    # Apply controls to the game
```

## Features

- **Normal Mode**: Neural network-based decision making
- **Demo Mode**: Manual demolition targeting (toggle with 'C' key)
- **Kickoff Sequences**: Hardcoded optimal kickoff patterns
- **Physics Prediction**: Advanced motion prediction for opponents
- **Attention Visualization**: Render neural network attention weights

## Testing

Run the structure validation tests:

```bash
python test_structure.py
```

This validates that all modules work correctly together and the refactoring preserved functionality.

## Dependencies

- `numpy` - For numerical computations
- `torch` - For neural network operations (when available)
- `rlbot` - For Rocket League bot framework (when available)
- `keyboard` - For manual mode switching
- `rlgym_compat` - For game state management

## Future Improvements

The modular structure makes it easy to:
- Add new game modes
- Improve physics calculations
- Enhance AI decision making
- Add new visualization features
- Implement better testing coverage