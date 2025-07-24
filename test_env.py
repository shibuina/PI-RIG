import gymnasium as gymnasium
import gymnasium_robotics

gymnasium.register_envs(gymnasium_robotics)

env = gymnasium.make('FetchPickAndPlaceDense-v3',render_mode='human')
env.reset()
for _ in range(2000):
    action = env.action_space.sample()
    env.step(action)
    env.render()
