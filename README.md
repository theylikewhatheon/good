# RLGym Ecosystem - Git Repository Structure Explanation

This repository contains a comprehensive collection of RLGym (Rocket League Gym) projects that work together to provide a complete reinforcement learning environment for the game Rocket League. This document explains the Git structure and codebase organization.

## Repository Structure

This Git repository contains three main components organized as subdirectories:

### 1. `rocket-league-gym-main/` - Core RLGym Library
The foundation of the ecosystem providing the core API and Rocket League environment implementation.

**Key Components:**
- **API Layer** (`rlgym/api/`): Zero-dependency base API for creating RL environments
- **Rocket League Implementation** (`rlgym/rocket_league/`): Specific implementation for Rocket League
- **Mathematical Utilities** (`rlgym/rocket_league/math.py`): Vector operations, rotations, and physics calculations
- **Setup Scripts**: Multiple setup files for different package configurations

**Mathematical Utilities Include:**
- Euclidean distance calculations
- Vector and scalar projections  
- Quaternion to Euler angle conversions
- Cosine similarity computations
- Vector normalization and magnitude functions

### 2. `rlgym-ppo-main/` - PPO Implementation
A vectorized Proximal Policy Optimization (PPO) implementation specifically designed for RLGym environments.

**Key Features:**
- Vectorized PPO algorithm for efficient training
- Support for both discrete and continuous action spaces
- Experience buffer management
- GPU acceleration support via PyTorch
- Batched agent management for multi-agent scenarios

### 3. `rlgym-tools-main/` - Extended Tools and Utilities
Additional tools and utilities that extend RLGym's functionality.

**Provides:**
- Extended action parsers
- Additional reward functions
- Custom observation builders
- State mutators for environment modification
- Replay file parsing utilities
- Community-contributed enhancements

## Git Workflow and Version Management

### Version Control Structure
The repository uses a standard Git structure with:
- **Main branch**: Contains stable releases and major updates
- **Feature branches**: Individual components and improvements
- **Commit history**: Tracks changes across all three major components

### Version Management System
Each component maintains its own versioning:

```python
# From rocket-league-gym-main/rlgym/version/version.py
__version__ = '2.0.0'

# From rocket-league-gym-main/rlgym/api/version.py  
__version__ = '2.0.0'

# From rocket-league-gym-main/rlgym/rocket_league/version.py
__version__ = '2.0.0'
```

### Setup and Package Management
The repository includes sophisticated setup scripts that handle:
- **Dependency management**: Different install options (api-only, full, simulation-only)
- **Package discovery**: Automatic namespace package detection
- **Version synchronization**: Ensures compatibility between components

Example from `setup_rlgym.py`:
```python
# Dynamic version loading
exports = {}
exec(open('rlgym/version/version.py').read(), exports)
version = exports['__version__']

# Dependency management with version constraints
requires = [
    'rlgym-api =={}'.format(api_version),
]

extras = {
    'rl': ['rlgym-rocket-league[all] =={}'.format(rl_version)],
    'rl-sim': ['rlgym-rocket-league[sim] =={}'.format(rl_version)],
}
```

## Installation and Usage

### Standard Installation
```bash
# Install complete ecosystem
pip install rlgym[all]

# Install only core API
pip install rlgym

# Install with Rocket League simulation
pip install rlgym[rl-sim]
```

### Development Installation
```bash
# Clone repository
git clone [repository-url]

# Install in development mode
cd good/
pip install -e rocket-league-gym-main/
pip install -e rlgym-ppo-main/  
pip install -e rlgym-tools-main/rlgym-tools-main/
```

## Code Organization Principles

### Modular Architecture
- **Separation of Concerns**: Each directory handles a specific aspect (core API, PPO, tools)
- **Namespace Packages**: Uses Python namespace packages for clean imports
- **Plugin Architecture**: Tools and extensions can be added without modifying core

### Configuration Management
The codebase uses JSON-based configuration management:
```python
# From setup.py
with open('setup.json', 'r') as setup_json:
    config = json.loads(setup_json.read())
```

### Example Integration
The `example.py` demonstrates how all components work together:
```python
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.rocket_league.sim import RocketSimEngine

# Complete environment setup
env = RLGym(
    obs_builder=DefaultObs(),
    action_parser=LookupTableAction(),
    reward_fn=CombinedReward(...),
    transition_engine=RocketSimEngine(),
)
```

## Contributing to the Repository

### Git Best Practices
1. **Feature Branches**: Create separate branches for new features
2. **Commit Messages**: Use descriptive commit messages
3. **Component Changes**: Keep changes focused to specific components when possible
4. **Version Updates**: Update version files when making breaking changes

### Testing
Each component includes its own testing framework:
- Unit tests for mathematical functions
- Integration tests for environment functionality  
- Example scripts for validation

## Technical Details

### Mathematical Foundation
The `math.py` module provides essential mathematical operations:
- **Vector Operations**: Projection, normalization, magnitude calculations
- **Quaternion Math**: Conversion between quaternion and Euler representations
- **Distance Metrics**: Euclidean distance and cosine similarity
- **Random Utilities**: Random vector generation with constraints

### Action Space Management
Support for multiple action space types:
- **Discrete Actions**: Finite set of possible actions
- **Continuous Actions**: Real-valued action vectors
- **Multi-Discrete**: Multiple discrete action dimensions
- **Lookup Tables**: Predefined action mappings

## Conclusion

This repository represents a complete ecosystem for Rocket League reinforcement learning, with careful attention to modularity, version management, and extensibility. The Git structure allows for independent development of components while maintaining integration capabilities.

For more detailed information about specific components, refer to the individual README files in each subdirectory.