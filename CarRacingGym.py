import gymnasium as gym
from stable_baselines3 import PPO


for i in range(10):
    # Initialize the environment with human rendering
    env = gym.make("CarRacing-v3", render_mode="human")

    # Load the saved model
    model = PPO.load("car_racing_logs/best_model.zip")


    # Reset the environment
    obs, info = env.reset()  # Extract observation from the reset tuple
    done = False
    while not done:
        # Predict the action
        action, _states = model.predict(obs, deterministic=True)
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        # Combine `terminated` and `truncated` to define `done`
        done = terminated or truncated

    env.close()
