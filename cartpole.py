import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment
env = gym.make("CartPole-v1")

# 2. Create the RL model (PPO algorithm)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the agent
model.learn(total_timesteps=10000)

# 4. Save the model
model.save("cartpole_model")

# 5. Watch the trained agent
obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
