# Git Workflow Guide for RLGym Ecosystem

This document provides specific Git commands and workflows for working with this RLGym ecosystem repository.

## Repository Overview

This repository contains multiple related projects in a single Git repository (monorepo structure):
- `rocket-league-gym-main/` - Core library
- `rlgym-ppo-main/` - PPO implementation  
- `rlgym-tools-main/` - Additional tools

## Git Commands for This Repository

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/theylikewhatheon/good.git
cd good

# View repository structure
git ls-tree -r --name-only HEAD

# Check repository status
git status

# View commit history
git log --oneline --graph --decorate
```

### Working with Multiple Components

Since this repository contains multiple projects, you'll often need to work across different directories:

```bash
# View changes across all components
git diff

# View changes in specific component
git diff rocket-league-gym-main/

# Add changes from specific component
git add rocket-league-gym-main/
git add rlgym-ppo-main/
git add rlgym-tools-main/

# Check which files are tracked
git ls-files | grep -E "(rocket-league-gym|rlgym-ppo|rlgym-tools)"
```

### Branch Management

```bash
# Create feature branch for specific component
git checkout -b feature/rlgym-core-update
git checkout -b feature/ppo-enhancement
git checkout -b feature/tools-addition

# List all branches
git branch -a

# Switch between branches
git checkout main
git checkout feature/rlgym-core-update

# Merge feature branch
git checkout main
git merge feature/rlgym-core-update
```

### Version Management Workflow

This repository uses coordinated versioning across components:

```bash
# Check current versions
grep -r "__version__" */

# View version history
git log --oneline --grep="version"

# Tag releases (when appropriate)
git tag -a v2.0.0 -m "Release version 2.0.0"
git push origin v2.0.0

# List all tags
git tag -l
```

### Component-Specific Operations

#### Working on Core RLGym (`rocket-league-gym-main/`)
```bash
# Navigate to core directory
cd rocket-league-gym-main/

# Check setup files
ls setup*.py

# View mathematical utilities
git log --oneline -- rlgym/rocket_league/math.py

# Stage only core changes
git add rocket-league-gym-main/
git commit -m "Update core RLGym functionality"
```

#### Working on PPO Implementation (`rlgym-ppo-main/`)
```bash
# View PPO-specific changes
git diff rlgym-ppo-main/

# Check PPO learner modifications
git log --oneline -- rlgym-ppo-main/rlgym_ppo/learner.py

# Commit PPO changes
git add rlgym-ppo-main/
git commit -m "Enhance PPO training algorithm"
```

#### Working on Tools (`rlgym-tools-main/`)
```bash
# View tools structure
tree rlgym-tools-main/ || find rlgym-tools-main/ -type f

# Check tools history
git log --oneline -- rlgym-tools-main/

# Add new tool
git add rlgym-tools-main/
git commit -m "Add new reward function tool"
```

### Examining Repository History

```bash
# View overall repository history
git log --oneline --all --graph

# View changes by author
git log --author="username" --oneline

# View changes in last 10 commits
git log -10 --stat

# View specific file history
git log --follow -- rocket-league-gym-main/setup.py

# View changes between commits
git diff HEAD~1 HEAD
```

### Troubleshooting Git Issues

#### Large Repository Management
```bash
# Check repository size
du -sh .git/

# View largest files
git ls-files | xargs ls -la | sort -k5 -rn | head

# Clean up (if needed)
git gc --aggressive
```

#### Working with Submodules (if any are added)
```bash
# Initialize submodules
git submodule init
git submodule update

# Update all submodules
git submodule update --remote

# Clone with submodules
git clone --recursive https://github.com/theylikewhatheon/good.git
```

### Development Workflow

#### Feature Development
```bash
# 1. Create feature branch
git checkout -b feature/new-functionality

# 2. Make changes to specific component
# Edit files in rocket-league-gym-main/, rlgym-ppo-main/, or rlgym-tools-main/

# 3. Stage and commit changes
git add .
git commit -m "Descriptive commit message"

# 4. Push feature branch
git push origin feature/new-functionality

# 5. Create pull request (via GitHub interface)
```

#### Release Workflow
```bash
# 1. Update version numbers in each component
# Edit version.py files in each project

# 2. Update setup.py files if needed
git add rocket-league-gym-main/rlgym/*/version.py
git add */setup*.py

# 3. Commit version updates
git commit -m "Bump version to X.Y.Z"

# 4. Tag release
git tag -a vX.Y.Z -m "Release version X.Y.Z"

# 5. Push with tags
git push origin main --tags
```

### Collaboration Guidelines

#### Before Making Changes
```bash
# Always pull latest changes
git pull origin main

# Check for conflicts
git status
```

#### Making Clean Commits
```bash
# Stage specific files only
git add rocket-league-gym-main/specific-file.py

# Use descriptive commit messages
git commit -m "component: brief description of change

Longer explanation of what changed and why."

# Example good commit messages:
git commit -m "core: fix quaternion to euler conversion in math.py"
git commit -m "ppo: improve experience buffer memory efficiency"
git commit -m "tools: add new reward function for ball possession"
```

#### Code Review Process
```bash
# Create patch for review
git format-patch -1 HEAD

# View changes before committing
git diff --cached

# Amend last commit if needed
git commit --amend
```

## Integration with Development Tools

### IDE Integration
Most IDEs can recognize this as a multi-component repository:
- Configure separate project roots for each component
- Set up separate linting/testing for each directory
- Use Git integration to view changes across components

### CI/CD Considerations
```bash
# Check which components changed
git diff --name-only HEAD~1 HEAD | grep -E "^(rocket-league-gym|rlgym-ppo|rlgym-tools)"

# Run tests for specific components based on changes
if git diff --name-only HEAD~1 HEAD | grep -q "rocket-league-gym-main"; then
    echo "Core changes detected, running core tests"
fi
```

This workflow documentation helps developers understand how to effectively work with the multi-component structure of this RLGym ecosystem repository.