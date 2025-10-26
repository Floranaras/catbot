# CatBot: Reinforcement Learning for Dynamic Cat Catching

A Q-learning implementation that trains an autonomous agent to catch cats with varying behavioral patterns on an 8x8 grid environment.

## Project Overview

CatBot uses Q-learning, a model-free reinforcement learning algorithm, to adaptively learn optimal strategies for catching cats with different personalities. The agent learns entirely through environmental interaction without hardcoded behaviors, demonstrating the generalization capabilities of reinforcement learning.

## Project Structure
```
catbot/
├── bot.py                    # Main training and execution script
├── cat_env.py                # Gymnasium environment implementation
├── training.py               # Q-learning algorithm implementation
├── utility.py                # Helper functions for gameplay and visualization
├── play.py                   # Manual play mode for testing
├── demo_final_only.py        # Automated demo showing trained performance
├── demo_with_training.py     # Automated demo showing learning progress
├── requirements.txt          # Python dependencies
├── images/                   # Cat and agent sprites
└── specs/                    # Project specifications
```

## Requirements

- Python 3.12 or higher
- gymnasium
- pygame
- numpy

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Train and Test Bot

Train the bot on a specific cat and watch it play:
```bash
python3 bot.py --cat batmeow
```

Available cats: `batmeow`, `mittens`, `paotsin`, `peekaboo`, `squiddyboi`, `trainer`

### Manual Play Mode

Test the environment manually using arrow keys:
```bash
python3 play.py --cat mittens
```

Controls:
- Arrow keys: Move agent
- Q: Quit

### Automated Demonstrations

Show trained bot performance only:
```bash
python3 demo_final_only.py
```

Show learning progress during training:
```bash
python3 demo_with_training.py
```

## Algorithm Details

### Q-Learning Implementation

The agent uses temporal difference Q-learning with the following update rule:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

### Hyperparameters

- Learning rate (α): 0.15
- Discount factor (γ): 0.95
- Epsilon decay: 0.9995
- Training episodes: 5000
- Max steps per episode: 200

### Reward Structure

- Catch cat: +100.0
- Move closer: +10.0
- Move away: -5.0
- Same distance: -1.0
- Step penalty: -0.5

### State Representation

States are encoded as 4-digit integers (0-9999):
- First two digits: Agent position (row, column)
- Last two digits: Cat position (row, column)

Example: State 2305 represents agent at (2,3) and cat at (0,5)

## Cat Behaviors

### Known Cats

- **Batmeow**: Stationary, does not move
- **Mittens**: Random movement in all directions
- **Paotsin**: Evasive, runs away when chased
- **Peekaboo**: Teleports to edges when adjacent to player
- **Squiddyboi**: Jumps over player when threatened

### Trainer Cat

The `trainer` cat can be customized in `cat_env.py` to test custom behaviors. Modify the `TrainerCat.move()` method to implement new movement patterns.

## Performance Metrics

Training completes in under 20 seconds per cat. Expected performance on known cats:

| Cat | Avg Steps | Success Rate |
|-----|-----------|--------------|
| Batmeow | 14 | 100% |
| Mittens | 11 | 100% |
| Paotsin | 15 | 100% |
| Peekaboo | 26 | 100% |
| Squiddyboi | 18 | 100% |

## Development

### Rendering Options

Control visualization frequency during training:
```bash
# No rendering (fastest)
python3 bot.py --cat paotsin --render -1

# Render every 100 episodes
python3 bot.py --cat paotsin --render 100

# Render every 500 episodes
python3 bot.py --cat paotsin --render 500
```

### Modifying Training Algorithm

Edit `training.py` to adjust:
- Hyperparameters (learning rate, discount factor, epsilon decay)
- Reward structure
- Exploration strategy

Only modify code within designated TODO sections.

### Commit Message Format
Follow **Conventional Commits** standard:

| Type | Purpose | 
|------|---------|
| `feat` | Add a new feature (functions, logic) |
| `refactor` | Improve code without changing behavior | 
| `perf` | Optimize performance (faster loops, better memory) | 
| `style` | Formatting changes (indentation, comments) | 
| `test` | Add or update test cases | 
| `build` | Modify build files or compilation setup | 
| `docs` | Update README, specs, or comments | 
| `chore` | Non-code maintenance (renaming files) | 

**Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```
Use semantic commit messages following this format:
```
<type>: <subject>
<type>(<scope>): <subject>
```
## Technical Constraints

- Training must complete in under 20 seconds
- Agent has maximum 60 moves to catch each cat
- State space: 10,000 possible states
- Action space: 4 discrete actions (up, down, left, right)

## Implementation Notes


The Q-learning algorithm uses epsilon-greedy exploration with exponential decay. Distance-based rewards guide the agent toward the cat while maintaining adaptability to different behavioral patterns. The implementation generalizes across cat types without hardcoded strategies.

## Credits

Cat images sourced from DLSU PUSA cat directory. See `images/credits.txt` for detailed attribution.

## License

Academic project for educational purposes. Not for commercial redistribution.
