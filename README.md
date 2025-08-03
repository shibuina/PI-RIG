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

### Run the Variational Autoencoder (VAE) for representation learning:

```bash
python rlkit/examples/rig/pusher/rig.py
```

### Run the RL algorithm (e.g. SAC) using RIG:

```bash
python rlkit/examples/rig/pusher/oracle.py
```

These scripts use the visual-based `Pusher` environment to demonstrate representation learning and reinforcement learning using RIG (Representation Learning for Goal-Conditioned RL).

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