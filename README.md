# Physics-Informed RIG: Goal-Conditioned RL with Physics Constraints

This project implements **Physics-Informed RIG (PI-RIG)**, a novel approach that integrates physics constraints into goal-conditioned reinforcement learning using [RLKit](https://github.com/vitchyr/rlkit) and [Multiworld](https://github.com/vitchyr/multiworld). The methodology incorporates an **Enhanced P³-VAE** architecture with physics-informed goal sampling for improved robotic manipulation tasks.

## 🎯 Project Overview

### Core Methodology

**Physics-Informed RIG** follows a four-stage pipeline:

1. **Data Collection**: Random/exploratory policy interaction to collect visual trajectories
2. **Representation Learning**: Enhanced P³-VAE with integrated physics constraints  
3. **Physics-Informed Goal Sampling**: Goals filtered by physical validity and reachability
4. **RL Training**: Goal-conditioned policy training with physics-consistent goals

### Key Innovations

- **Enhanced P³-VAE**: Latent space decomposed into physics-intrinsic and environmental variables
- **Physics Constraints**: Momentum conservation, gravity modeling, temporal consistency
- **Semi-Supervised Learning**: Leverages ground truth physics states from MuJoCo simulator
- **Multi-Task Support**: Validated on pusher, pick-and-place, and reacher tasks

### Supported Robotic Tasks

- **Pusher**: Object manipulation with puck pushing
- **Pick-and-Place**: Object manipulation with Sawyer arm robot  
- **Reacher**: Point reaching with arm dynamics

---

## 🔧 Setup Instructions

1. Clone this repository and ensure you have `conda` installed.
2. Run the setup script:

```bash
bash setup.sh
```

This script will:
- Create a Conda environment named `final` from `env.yml`
- Install `rlkit` and `multiworld` using `pip install -e`
- Set up all necessary dependencies

---

## 🚀 Running Experiments

### Task-Specific Physics-Informed RIG

**Pusher Task:**
```bash
python rlkit/examples/rig/pusher/pi_rig.py
```

**Pick-and-Place Task:**
```bash
python rlkit/examples/rig/pick_and_place/pi_rig.py
```

**Reacher Task:**
```bash
python rlkit/examples/rig/reacher/pi_rig.py
```

### Baseline Comparisons

Each task directory contains comparison baselines:
- `rig.py`: Standard RIG without physics constraints
- `oracle.py`: Oracle with ground truth goals
- `skewfit.py`: SkewFit baseline
- `ccrig.py`: Curiosity-driven RIG variant
### Configuration Testing

Validate PI-RIG configuration for each task:


### Task-Specific Notes

**Important**: Update the system path (lines 5-7) in experiment files to match your project directory to resolve import issues.

**Pusher**: Manipulation with physics constraints on puck dynamics
**Pick-and-Place**: Manipulation with grasp stability and object physics  
**Reacher**: Point reaching with joint dynamics and momentum conservation

---

## 📊 Visualizing Results and Policies

### Viewing Training Results
Results are saved to task-specific directories:
```
rlkit/data/<task>/<timestamp>/
```

**Visualize training metrics:**
```bash
# Pusher results
python -m viskit.frontend rlkit/data/pusher_final/

# Pick-and-place results  
python -m viskit.frontend rlkit/data/compare_pick_and_place_final/

# Reacher results
python -m viskit.frontend rlkit/data/reacher_final/
```

### Policy Simulation
```bash
python rlkit/scripts/sim_policy.py rlkit/data/<task>/<timestamp>/params.pkl
```

---

## 📁 Project Structure

```
.
├── README.md                          # This file
├── setup.sh                          # Environment setup script  
├── env.yml                           # Conda environment specification
├── requirements.txt                  # Python package requirements
├── paper/
│   └── methodology.tex               # Detailed methodology description
├── rlkit/                           # Modified RLKit with PI-RIG components
│   ├── examples/rig/                # Task-specific experiments
│   │   ├── pusher/                  # Pusher manipulation
│   │   │   ├── pi_rig.py           # Physics-Informed RIG 
│   │   │   ├── rig.py              # Standard RIG baseline
│   │   │   ├── oracle.py           # Oracle baseline
│   │   │   ├── skewfit.py          # SkewFit baseline
│   │   │   └── ccrig.py            # Curiosity-driven RIG
│   │   ├── pick_and_place/         # 3D pick-and-place manipulation
│   │   │   ├── pi_rig.py           # Physics-Informed RIG
│   │   │   ├── rig.py              # Standard RIG baseline
│   │   │   ├── oracle.py           # Oracle baseline
│   │   │   ├── skewfit.py          # SkewFit baseline
│   │   │   └── ccrig.py            # Curiosity-driven RIG
│   │   └── reacher/                # 2-DOF reacher task
│   │       ├── pi_rig.py           # Physics-Informed RIG
│   │       ├── rig.py              # Standard RIG baseline
│   │       ├── oracle.py           # Oracle baseline
│   │       ├── skewfit.py          # SkewFit baseline
│   │       └── ccrig.py            # Curiosity-driven RIG
│   └── rlkit/torch/vae/            # Core VAE implementations
│       ├── enhanced_p3_vae_trainer.py     # Enhanced P³-VAE
│       ├── physics_informed_goal_sampling.py  # Goal sampling
│       └── vae_trainer.py          # Standard VAE trainer
├── multiworld/                     # Goal-conditioned environments
│   └── envs/                      # MuJoCo environments
└── viskit/                        # Training visualization tools
```

---

## 🔬 Key Features

### Enhanced P³-VAE Architecture
- **Latent Space Decomposition**: Physics-intrinsic (z_I) and environmental (z_E) variables
- **Physics-Guided Encoder**: CNN backbone with task-specific physics heads
- **Semi-Supervised Learning**: Leverages ground truth physics states from MuJoCo
- **Multi-Loss Training**: VAE reconstruction + physics consistency + conservation laws

### Physics Constraints Integration
- **Temporal Consistency**: Smooth latent state transitions across time
- **Momentum Conservation**: Enforces realistic dynamics in latent space
- **Task-Specific Physics**: 
  - Pusher: Collision physics
  - Pick-and-Place: Grasp stability, gravity
  - Reacher: Joint dynamics, angular momentum

### Physics-Informed Goal Sampling
- **Validity Filtering**: Goals filtered by physical plausibility P(z_g)
- **Reachability Assessment**: Goals evaluated for state-conditional reachability R(z_g|s_t)
- **Dynamic Adjustment**: Goal difficulty adapts during training

---

## References

- [RLKit (BAIR)](https://github.com/vitchyr/rlkit): Deep RL algorithms and utilities
- [Multiworld (BAIR)](https://github.com/vitchyr/multiworld): Goal-conditioned environments  
- [RIG Paper](https://arxiv.org/abs/1807.04742): "Self-Supervised Visual RL with Imagined Goals"

---

## 🤝 Contributing

For questions, issues, or contributions:
1. Check existing issues and documentation
2. Validate configuration with task-specific test scripts
3. Test on all three supported tasks (pusher, pick-and-place, reacher)

---

Feel free to reach out if you encounter any issues or want to extend this framework to other robotic tasks.
