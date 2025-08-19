# Physics-Informed RIG: Goal-Conditioned RL with Physics Constraints

This project implements **Physics-Informed RIG (Representation Learning for Goal-Conditioned RL)** using [RLKit](https://github.com/vitchyr/rlkit) and [Multiworld](https://github.com/vitchyr/multiworld). The main contribution is comparing physics-informed vs standard representation learning for robotic pick-and-place tasks.

## 🎯 Project Overview

This research compares two approaches to goal-conditioned reinforcement learning:

1. **Physics-Informed RIG**: Incorporates physics constraints (gravity, momentum, contact forces) into VAE representation learning
2. **Standard RIG**: Uses standard VAE representation learning without physics constraints

The comparison is performed on the **Sawyer Pick and Place** robotic manipulation task.

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

## 🚀 Main Experiments

### Physics-Informed vs Standard RIG Comparison

Run the main paper experiment comparing both approaches:
Note: before you run the code, please change your sys path (line 5-7 in code file) to your main folder directory to handle import problems
```bash
# Navigate to the pick and place experiments
cd rlkit/examples/rig/pick_and_place

# # Run VAE-only comparison 
# python physics_vs_standard_rig_comparison.py

# Run complete RIG training comparison (VAE + RL training)
python full_rig_training_comparison.py
```

Similarly, for pusher run: 
```bash
python rlkit/examples/rig/pusher/physics_informed_rig.py
```
### Visualization
The result will be saved in 2 folder with suffix *full-rig-physics and *full-rig-standard, you can copy the subfolder to a new folder to compare between 2 algorithm
Example:
```bash
python -m viskit.frontend rlkit/data/compare_pick_and_place_500ep/ 
```
For pusher, run:
```bash
python -m viskit.frontend rlkit/data/compare_pusher/
```
<!-- ### Quick Demo

For a simple demonstration of physics-informed representation learning:

```bash
cd rlkit/examples/rig/pick_and_place
python simple_physics_comparison.py
```

--- -->

## 📁 Project Structure

```
.
├── README.md                           # This file
├── setup.sh                           # Environment setup script
├── env.yml                            # Conda environment specification
├── requirements.txt                   # Python package requirements
├── rlkit/                             # Modified RLKit with physics-informed components
│   ├── examples/rig/pick_and_place/   # Main pick-and-place experiments
│   │   ├── physics_vs_standard_rig_comparison.py    # VAE comparison (working)
│   │   ├── full_rig_training_comparison.py          # Complete RIG comparison
│   │   ├── simple_physics_comparison.py             # Quick demo
│   │   ├── physics_informed_rig.py                  # Core implementation
│   │   └── README.md                               # Detailed experiment guide
│   ├── rlkit/torch/vae/               # VAE implementations
│   │   ├── pick_and_place_physics.py  # Physics-informed VAE trainer
│   │   └── vae_trainer.py             # Standard VAE trainer
│   └── rlkit/torch/grill/             # GRILL framework integration
│       └── common.py                  # Common VAE training utilities
│
├── multiworld/                        # Goal-conditioned environments
│   └── envs/mujoco/sawyer_xyz/        # Sawyer robot environments
└── viskit/                           # Visualization tools
```

---

## 🔬 Key Features

### Physics-Informed Representation Learning
- **Temporal Consistency**: Ensures smooth latent transitions
- **Momentum Conservation**: Enforces physics-based momentum constraints  
- **Gravity Modeling**: Incorporates gravitational effects
- **Contact Forces**: Models object-object and robot-object interactions
- **Grasp Stability**: Ensures realistic grasping physics

### Comparison Metrics
- **Success Rates**: Task completion rates for pick-and-place
- **Sample Efficiency**: Learning speed and data requirements
- **Generalization**: Performance on unseen object configurations
- **Representation Quality**: Latent space interpretability and structure

---

## 📊 Expected Results

The physics-informed approach should demonstrate:
- **Higher success rates** on pick-and-place tasks
- **Better generalization** to new object configurations  
- **More sample-efficient** RL training
- **More interpretable** learned representations
- **Improved stability** in manipulation policies

---

## 🛠 Development Notes

This project extends the original RLKit and Multiworld frameworks with:
- Custom physics-informed VAE trainers
- Enhanced loss functions incorporating physics constraints
- Robotic manipulation-specific physics modeling
- Comprehensive comparison and evaluation tools

---

## 📖 References

- [RLKit](https://github.com/vitchyr/rlkit): Deep RL algorithms and utilities
- [Multiworld](https://github.com/vitchyr/multiworld): Goal-conditioned environments
- Original RIG paper: "Hindsight Experience Replay for Image-Based Robotic Manipulation"

---

## 📚 References

- [RLKit (BAIR)](https://github.com/vitchyr/rlkit)
- [Multiworld (BAIR)](https://github.com/vitchyr/multiworld)
- [RIG: Self-Supervised Visual RL with Imagined Goals](https://arxiv.org/abs/1807.04742)

---

Feel free to reach out if you encounter any issues or want to extend this setup to other environments or algorithms.