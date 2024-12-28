import gymnasium as gym
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO

# Parameters
ENV_NAME = "CarRacing-v3"
NUM_EPISODES = 10       # Number of episodes
EPISODE_LENGTH = 1000   # Steps per episode
SAVE_INTERVAL = 1      # Save an image every 10 steps
OUTPUT_DIR = "MDN-RNN"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create the environment in 'rgb_array' mode so env.render() returns images
env = gym.make(ENV_NAME, render_mode="rgb_array")

# Load the saved model
model = PPO.load("car_racing_logs/best_model.zip")

for episode in range(1, NUM_EPISODES + 1):
    # Create a subdirectory for this episode
    episode_dir = os.path.join(OUTPUT_DIR, f"episode_{episode}")
    os.makedirs(episode_dir, exist_ok=True)

    # Open a text file for logging actions in this episode
    actions_log_path = os.path.join(episode_dir, "actions.txt")
    with open(actions_log_path, "w") as f_log:
        f_log.write("Step,Action\n")  # Header line for clarity

    # Reset the environment
    obs, info = env.reset()
    print(f"Starting Episode {episode}")

    # Step through the episode
    for step in range(EPISODE_LENGTH):
        # Render returns an RGB array (H x W x 3)
        frame = env.render()

        # Predict the action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Save the current frame and log the action every SAVE_INTERVAL steps
        if step % SAVE_INTERVAL == 0:
            # Construct image filename
            filename = os.path.join(episode_dir, f"Img_step_{step}.png")
            # Save the frame
            plt.imsave(filename, frame)
            # Log the action
            with open(actions_log_path, "a") as f_log:
                f_log.write(f"{step},{action}\n")

            print(f"Episode {episode}, Step {step}: Saved snapshot to {filename} with action {action}")

        # End episode if done or truncated
        if done or truncated:
            print(f"Episode {episode} ended at step {step}.")
            break

env.close()
print("All episodes complete.")
