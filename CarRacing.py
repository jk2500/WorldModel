import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os

# Set up the environment
env_name = "CarRacing-v3"

def make_env():
    """Helper function to create a monitored environment."""
    env = gym.make(env_name, continuous=True)
    return Monitor(env)

# Number of parallel environments
num_envs = 4  # Adjust based on your system's capabilities

# Create vectorized environments
env = make_vec_env(make_env, n_envs=num_envs)

# Set up directory to save the model
log_dir = "./car_racing_logs/"
os.makedirs(log_dir, exist_ok=True)

# Define a callback to stop training when reward threshold is reached
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=callback_on_best,
    verbose=1,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
)

# Instantiate the PPO model
model = PPO(
    "CnnPolicy",  # CNN policy to handle image inputs
    env,
    verbose=1,
    tensorboard_log=log_dir,
    device="mps",  # Explicitly use M1 GPU via Metal
)

# Train the model
training_timesteps = 500000  # Adjust based on available compute
model.learn(total_timesteps=training_timesteps, callback=eval_callback)

# Save the trained model
model_path = os.path.join(log_dir, "ppo_car_racing_model.zip")
model.save(model_path)

# Load and test the trained model
print("Training complete. Testing the model...")
test_env = gym.make(env_name, continuous=True, render_mode=None)  # Optimize rendering for testing
obs = test_env.reset()
done = False
frame_skip = 5  # Render every nth frame
frame_count = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    frame_count += 1
    if frame_count % frame_skip == 0:
        test_env.render()

test_env.close()
