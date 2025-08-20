# Git Code Structure - Quick Reference

This is a quick reference for understanding the Git structure and key commands for the RLGym ecosystem repository referenced as @Khagendra01/Python-MLL.

## Repository Structure at a Glance

```
.
├── .git/                    # Git metadata and history
├── rocket-league-gym-main/  # Core RLGym library
│   ├── rlgym/
│   │   ├── api/            # Base API (zero dependencies)
│   │   ├── rocket_league/  # RL-specific implementation
│   │   └── version/        # Version management
│   ├── setup.py           # Main package setup
│   ├── setup_api.py       # API-only setup
│   ├── setup_rlgym.py     # Full setup configuration
│   └── setup_rocket_league.py # RL-specific setup
├── rlgym-ppo-main/         # PPO implementation
│   ├── rlgym_ppo/         # PPO algorithm code
│   ├── example.py         # Usage example
│   └── various utilities
└── rlgym-tools-main/       # Extended tools and utilities
    └── rlgym-tools-main/
        ├── rlgym_tools/   # Tool implementations
        ├── setup.py       # Tools package setup
        └── tests/         # Test suite
```

## Key Git Commands for This Repository

### Repository Inspection
```bash
# View the current repository structure
git ls-tree -r --name-only HEAD

# Check recent commits
git log --oneline -10

# See what's in the staging area
git status

# View all branches
git branch -a
```

### Current Repository State
Based on the commit history:
```bash
git log --oneline
# 8db9bc8 (HEAD) Initial plan
# 2df1b47 Add files via upload
```

### Working with Components

#### Check changes in specific components:
```bash
# Core library changes
git diff rocket-league-gym-main/

# PPO implementation changes  
git diff rlgym-ppo-main/

# Tools changes
git diff rlgym-tools-main/
```

#### Stage specific components:
```bash
# Add only core library changes
git add rocket-league-gym-main/

# Add only PPO changes
git add rlgym-ppo-main/

# Add only tools changes
git add rlgym-tools-main/
```

## Understanding the Setup Scripts

This repository contains multiple setup.py files for different installation options:

### `rocket-league-gym-main/setup.py`
- Uses `setup.json` for configuration
- Dynamically loads base configuration
- Merges with JSON config for final setup

### `rocket-league-gym-main/setup_rlgym.py`
- Creates the main `rlgym` package
- Manages dependencies between api, core, and RL components
- Generates `setup.json` for the main setup.py

### `rocket-league-gym-main/setup_api.py`
- Creates `rlgym-api` package (zero dependencies)
- Standalone API package

### `rocket-league-gym-main/setup_rocket_league.py`
- Creates `rlgym-rocket-league` package
- Includes simulation and visualization dependencies

## Version Management

Each component maintains its version:
```bash
# Check all version files
find . -name "version.py" -exec grep -H "__version__" {} \;

# Current versions (as of repository state):
# rocket-league-gym-main/rlgym/version/version.py: __version__ = '2.0.0'
# rocket-league-gym-main/rlgym/api/version.py: __version__ = '2.0.0'  
# rocket-league-gym-main/rlgym/rocket_league/version.py: __version__ = '2.0.0'
```

## Key Files to Understand

### Mathematical Utilities (`rocket-league-gym-main/rlgym/rocket_league/math.py`)
Core mathematical functions for 3D operations:
- `euclidean_distance()` - Distance between vectors
- `vector_projection()` - Vector projection operations
- `scalar_projection()` - Scalar projection
- `quat_to_euler()` - Quaternion to Euler angle conversion
- `normalize()` - Vector normalization
- `cosine_similarity()` - Cosine similarity between vectors

### Example Usage (`rocket-league-gym-main/example.py`)
Demonstrates integration of all components:
- Environment creation with `RLGym()`
- Action parsing with `LookupTableAction()`
- Reward functions with `CombinedReward()`
- State management and rendering

### PPO Implementation (`rlgym-ppo-main/rlgym_ppo/learner.py`)
Main learning algorithm implementation for training RL agents.

## Git Workflow for This Repository

### Making Changes
```bash
# 1. Create a feature branch
git checkout -b feature/description

# 2. Make changes to specific component(s)
# Edit files in rocket-league-gym-main/, rlgym-ppo-main/, or rlgym-tools-main/

# 3. Stage changes
git add <component-directory>/

# 4. Commit with descriptive message
git commit -m "component: description of changes"

# 5. Push branch
git push origin feature/description
```

### Reviewing Changes
```bash
# See what changed in last commit
git show --stat

# Compare with previous version
git diff HEAD~1

# View file history
git log --oneline --follow <file-path>
```

## Installation from Git Repository

### Development Installation
```bash
# Clone repository
git clone <repository-url>
cd good

# Install components in development mode
pip install -e rocket-league-gym-main/
pip install -e rlgym-ppo-main/
pip install -e rlgym-tools-main/rlgym-tools-main/
```

### Understanding Dependencies
The setup scripts define complex dependency relationships:
- `rlgym-api` has zero dependencies
- `rlgym-rocket-league` depends on `rlgym-api` and numpy
- `rlgym` (main package) coordinates all components
- Optional extras for simulation (`[sim]`) and visualization (`[rlviser]`)

## Common Git Operations

### Checking Repository Health
```bash
# Repository size
du -sh .git/

# Number of commits
git rev-list --count HEAD

# Contributors
git shortlog -sn

# Files tracked
git ls-files | wc -l
```

### Understanding the Codebase
```bash
# Find Python files
find . -name "*.py" | head -20

# Search for specific functionality
git grep -n "quaternion" 
git grep -n "PPO"
git grep -n "reward"

# View package structure
find . -name "__init__.py" -exec dirname {} \;
```

This quick reference provides the essential Git commands and understanding needed to work with this RLGym ecosystem repository.