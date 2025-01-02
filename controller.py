import torch
import numpy as np
from torchvision import transforms as T
from VAE import VAE
from MDNRNN import MDNRNN
import gymnasium as gym
from cma import CMAEvolutionStrategy
import os
import multiprocessing
from functools import partial

# === Set Hyperparameters ===
z_dim = 1024  # Latent dimension size for VAE
hidden_dim = 256  # Hidden state size for MDNRNN
action_dim = 3  # Number of action dimensions (steering, gas, brake)
num_mixtures = 5  # Number of mixture components for MDNRNN
controller_population_size = 12  # Number of controllers in CMA-ES population
mdn_temperature = 2.0  # Temperature parameter for MDNRNN

# Transformation for gym environment RGB output
transform = T.Compose([
    T.ToPILImage(),       # Convert np.ndarray to PIL Image
    T.Resize((64, 64)),   # Resize image to 64x64 pixels
    T.ToTensor()          # Convert PIL Image to torch.Tensor
])

def load_vae_model(filepath):
    """Load a pretrained VAE model from the given filepath."""
    vae_model = VAE(latent_dim=z_dim)
    vae_model.load_state_dict(torch.load(filepath, map_location="cpu"))
    vae_model.eval()  # Set model to evaluation mode
    return vae_model

def load_mdn_rnn_model(filepath):
    """Load a pretrained MDNRNN model from the given filepath."""
    mdn_rnn_model = MDNRNN(z_dim=z_dim, action_dim=action_dim,
                           hidden_dim=hidden_dim, num_mixtures=num_mixtures)
    mdn_rnn_model.load_state_dict(torch.load(filepath, map_location="cpu"))
    mdn_rnn_model.eval()  # Set model to evaluation mode
    return mdn_rnn_model

def preprocess_rgb_frame(frame):
    """
    Preprocess an RGB frame from the environment.

    Parameters:
        frame (np.ndarray): Input frame of shape (height, width, 3) with uint8 values [0..255].

    Returns:
        torch.Tensor: Preprocessed frame of shape (1, 3, 64, 64) with values in [0..1].
    """
    frame_tensor = transform(frame)  # Transform frame to tensor with resizing
    return frame_tensor.unsqueeze(0)  # Add batch dimension

def evaluate_controller(controller_weights, vae_filepath, mdn_rnn_filepath):
    """
    Evaluate a single controller on a new environment instance.

    Parameters:
        controller_weights (np.ndarray): Weights for the controller.
        vae_filepath (str): Path to the pretrained VAE model.
        mdn_rnn_filepath (str): Path to the pretrained MDNRNN model.

    Returns:
        float: Cumulative reward achieved by the controller.
    """
    # Initialize the environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # Load pretrained models
    vae = load_vae_model(vae_filepath)
    mdn_rnn = load_mdn_rnn_model(mdn_rnn_filepath)

    cumulative_reward = 0  # Track total reward
    obs, _ = env.reset()  # Reset environment
    h, c = None, None  # Initialize hidden states

    # Reshape controller weights into a matrix for actions
    weight_matrix = controller_weights.reshape(action_dim, -1)

    done = False
    while not done:
        # Render the environment and preprocess the frame
        frame = env.render()
        frame_preprocessed = preprocess_rgb_frame(frame)

        # Encode the frame to a latent vector (z)
        with torch.no_grad():
            z, _ = vae.encoder(frame_preprocessed)

        # Flatten latent vector and hidden state
        z_np = z.cpu().numpy().flatten()
        h_np = h.cpu().numpy().flatten() if h is not None else np.zeros(hidden_dim)

        # Combine latent vector and hidden state to form input
        controller_input = np.hstack([z_np, h_np])

        # Compute action using the controller weights
        action = weight_matrix @ controller_input

        # Clip action values to valid ranges
        action[0] = np.clip(action[0], -1., 1.)  # Steering
        action[1] = np.clip(action[1],  0., 1.)  # Gas
        action[2] = np.clip(action[2],  0., 1.)  # Brake

        # Step the environment with the computed action
        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward

        # Prepare inputs for MDNRNN
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        z_seq = z.unsqueeze(1)

        # Initialize hidden states if not already done
        if h is None or c is None:
            num_layers = 1  # Number of RNN layers
            batch_size = z_seq.size(0)
            h = torch.zeros(num_layers, batch_size, hidden_dim)
            c = torch.zeros(num_layers, batch_size, hidden_dim)

        # Forward pass through MDNRNN
        with torch.no_grad():
            _, _, _, (h, c) = mdn_rnn.forward(z_seq, action_tensor, (h, c),
                                             temperature=mdn_temperature)

        # Check if episode is done
        done = terminated or truncated

    env.close()  # Close the environment
    return cumulative_reward

def optimize_controller(vae_filepath, mdn_rnn_filepath,
                        population_size=controller_population_size):
    """
    Optimize the controller using CMA-ES.

    Parameters:
        vae_filepath (str): Path to the pretrained VAE model.
        mdn_rnn_filepath (str): Path to the pretrained MDNRNN model.
        population_size (int): Size of the population for CMA-ES.

    Returns:
        np.ndarray: Weights of the best controller.
    """
    # Number of parameters for the controller
    num_params = (z_dim + hidden_dim) * action_dim

    # Initialize CMA-ES
    es = CMAEvolutionStrategy(np.zeros(num_params), 0.5,
                              {"popsize": population_size})

    # Parallelism settings
    n_workers = population_size
    best_score = -np.inf  # Track best score
    best_weights = None  # Store best weights
    recent_scores = []  # Track recent rewards for stopping criterion

    # Directory to save weights
    save_directory = "controller_weights"
    os.makedirs(save_directory, exist_ok=True)

    with multiprocessing.Pool(processes=n_workers) as pool:
        while not es.stop():
            # Generate candidate solutions
            solutions = es.ask()

            # Evaluate solutions in parallel
            eval_func = partial(evaluate_controller, 
                                vae_filepath=vae_filepath, 
                                mdn_rnn_filepath=mdn_rnn_filepath) 

            rewards = pool.map(eval_func, solutions) 

            # Compute fitness (negative reward for minimization)
            fitness = [-r for r in rewards] 

            # Update CMA-ES with fitness values
            es.tell(solutions, fitness) 

            # Track the best solution
            best_idx = np.argmin(fitness)
            current_best_score = rewards[best_idx]

            if current_best_score > best_score:
                best_score = current_best_score
                best_weights = solutions[best_idx]

                # Save new best weights
                save_path = os.path.join(save_directory, "best_controller_weights.npy")
                np.save(save_path, best_weights)
                print(f"New best score: {best_score:.2f}. Weights saved to {save_path}")

            # Add the best score of the current generation to recent scores
            recent_scores.append(current_best_score)
            if len(recent_scores) > 5:
                recent_scores.pop(0)  # Keep only the last 5 scores

            # Check stopping criterion based on coefficient of variation
            if len(recent_scores) == 5:
                mean_score = np.mean(recent_scores)
                std_dev = np.std(recent_scores)
                if mean_score != 0 and (std_dev / mean_score) * 100 < 3:
                    print(f"Stopping early as coefficient of variation is less than 3%. Best score: {best_score:.2f}")
                    break

            print(f"Best Reward this gen: {rewards[best_idx]:.2f}")

        return best_weights

def main(): 
    """Main function to optimize and evaluate the controller."""
    vae_filepath = "VAE_1.pth"  # Path to VAE model
    mdn_rnn_filepath = "mdn_rnn.pth"  # Path to MDNRNN model

    # Optimize the controller
    best_controller_weights = optimize_controller(vae_filepath, mdn_rnn_filepath) 
    print("Controller Optimization Complete.") 

    # Evaluate the optimized controller
    final_reward = evaluate_controller(best_controller_weights,
                                       vae_filepath,
                                       mdn_rnn_filepath)
    print(f"Final Test Reward: {final_reward:.2f}")

if __name__ == "__main__":
    main()
