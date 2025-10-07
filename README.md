# RIG + RLKit + Multiworld Setup

This project uses [RLKit](https://github.com/vitchyr/rlkit) and [Multiworld](https://github.com/vitchyr/multiworld), originally developed by the Berkeley Artificial Intelligence Research (BAIR) Lab. The environment is configured to support learning visual representations and Reinforcement Learning (RL) with goal-conditioned tasks.

---

## 🔧 Setup Instructions

1. Clone this repository and ensure you have `conda` installed.
2. Run the setup script:

```bash
bash setup.sh
```

This script will:

- Check if the Conda environment named `final` exists; if not, it creates one from `env.yml`.
- Install `rlkit` and `multiworld` into the environment using `pip install -e`.

---

## 🚀 Running the Code

### Run the Physics-Informed Variational Autoencoder (VAE) for representation learning:

```bash
python rlkit/examples/rig/pusher/physics_informed_rig.py
```

### Run the Base Reinforcement learning based on Imaginary Goals (RIG):

```bash
python rlkit/examples/rig/pusher/physics_informed_rig.py
```

### Run the RL algorithm (e.g. SAC) using RIG:

```bash
python rlkit/examples/rig/pusher/oracle.py
```

These scripts use the visual-based `Pusher` environment to demonstrate representation learning and reinforcement learning using RIG (Representation Learning for Goal-Conditioned RL) with physics-informed constraints.

---

## 📊 Visualizing Results and Policies

### Viewing Training Results
During training, results are saved to:
```
rlkit/data/<exp_prefix>/<foldername>/
```

To visualize training progress and metrics, use viskit:
```bash
python viskit/viskit/frontend.py rlkit/data/<exp_prefix>/
```

For a single experiment:
```bash
python viskit/viskit/frontend.py rlkit/data/<exp_prefix>/<foldername>/
```

### Visualizing Trained Policies
```bash
python rlkit/scripts/sim_policy.py rlkit/data/<exp_prefix>/<foldername>/params.pkl
```
**Example:**
After running `physics_informed_rig.py`, you might find results in a folder like:
```
rlkit/data/pusher-physics-rig/2025_10_05_12_34_56_000000--s-0/
```

Then visualize with:
```bash
python rlkit/scripts/sim_policylicy.py rlkit/data/pusher-physics-rig/2025_10_05_12_34_56_000000--s-0/params.pkl
```

---

## 📁 Project Structure

```
.
├── env.yml                # Environment specification
├── rlkit/                 # RL algorithms and RIG implementation
├── multiworld/            # Goal-conditioned environment
├── setup.sh               # Environment and dependency setup script
└── README.md              # This file
```

---

## 📚 References

- [RLKit (BAIR)](https://github.com/vitchyr/rlkit)
- [Multiworld (BAIR)](https://github.com/vitchyr/multiworld)
- [RIG: Self-Supervised Visual RL with Imagined Goals](https://arxiv.org/abs/1807.04742)

---

Feel free to reach out if you encounter any issues or want to extend this setup to other environments or algorithms.