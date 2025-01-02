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
z_dim = 1024
hidden_dim = 256
action_dim = 3
num_mixtures = 5
controller_population_size = 12
mdn_temperature = 2.0

# Transformation for gym environment RGB output
transform = T.Compose([
    T.ToPILImage(),       # convert np.ndarray to PIL Image
    T.Resize((64, 64)),   # resize to 64x64
    T.ToTensor()          # convert PIL Image to torch.Tensor
])


def load_vae_model(filepath):
    vae_model = VAE(latent_dim=z_dim)
    vae_model.load_state_dict(torch.load(filepath, map_location="cpu",  weights_only=True))
    vae_model.eval()
    return vae_model


def load_mdn_rnn_model(filepath):
    mdn_rnn_model = MDNRNN(z_dim=z_dim, action_dim=action_dim,
                           hidden_dim=hidden_dim, num_mixtures=num_mixtures)
    mdn_rnn_model.load_state_dict(torch.load(filepath, map_location="cpu", weights_only=True))
    mdn_rnn_model.eval()
    return mdn_rnn_model


def preprocess_rgb_frame(frame):
    """
    frame is a NumPy array of shape (height, width, 3) with uint8 values [0..255].
    This function will return a torch.Tensor of shape (1, 3, 64, 64) in [0..1].
    """
    frame_tensor = transform(frame)     # shape (3,64,64)
    return frame_tensor.unsqueeze(0)    # shape (1,3,64,64)


def evaluate_controller(controller_weights, vae_filepath, mdn_rnn_filepath):
    """
    Evaluate a single controller (weights) on a brand-new environment.
    We load models (VAE, MDN-RNN) on CPU to avoid GPU concurrency issues.
    """
    # 1. Create a separate environment per process
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # 2. Load VAE and MDNRNN (CPU only for concurrency safety)
    vae = load_vae_model(vae_filepath)
    mdn_rnn = load_mdn_rnn_model(mdn_rnn_filepath)

    cumulative_reward = 0
    obs, _ = env.reset()
    h, c = None, None

    # Reshape controller_weights to produce 3D action
    weight_matrix = controller_weights.reshape(action_dim, -1)

    done = False
    while not done:
        frame = env.render()  # shape (window_h, window_w, 3)
        frame_preprocessed = preprocess_rgb_frame(frame)

        # Encode frame to latent z
        with torch.no_grad():
            z, _ = vae.encoder(frame_preprocessed)

        # Convert z to 1D numpy array for the linear controller
        z_np = z.cpu().numpy().flatten()

        # If h is None, create zeros for the controller
        if h is not None:
            h_np = h.cpu().numpy().flatten()
        else:
            h_np = np.zeros(hidden_dim)

        # Compute action via linear controller (NumPy)
        controller_input = np.hstack([z_np, h_np])  # shape: (z_dim+hidden_dim,)
        action = weight_matrix @ controller_input   # shape: (3,)

        # Clip / squash actions
        action[0] = np.clip(action[0], -1., 1.)  # steering
        action[1] = np.clip(action[1],  0., 1.)  # gas
        action[2] = np.clip(action[2],  0., 1.)  # brake

        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward

        # Convert action to torch.Tensor for MDNRNN
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        z_seq = z.unsqueeze(1)  # shape: (1, 1, z_dim)

        if h is None or c is None:
            num_layers = 1  # Adjust if MDNRNN uses more layers
            batch_size = z_seq.size(0)  # likely 1
            h = torch.zeros(num_layers, batch_size, hidden_dim)
            c = torch.zeros(num_layers, batch_size, hidden_dim)

        with torch.no_grad():
            # Forward pass through MDNRNN
            _, _, _, (h, c) = mdn_rnn.forward(z_seq, action_tensor, (h, c),
                                             temperature=mdn_temperature)

        done = terminated or truncated

    env.close()
    return cumulative_reward


def optimize_controller(vae_filepath, mdn_rnn_filepath,
                        population_size=controller_population_size):
    """
    Parallelize the CMA-ES evaluation step. Each solution in the population
    is evaluated in its own process with a fresh environment.
    """
    # We want a linear mapping from (z_dim + hidden_dim) -> 3
    num_params = (z_dim + hidden_dim) * action_dim

    es = CMAEvolutionStrategy(np.zeros(num_params), 0.5,
                              {"popsize": population_size})

    # Prepare a pool of workers
    n_workers = population_size  # or more, if you have more CPU cores
    with multiprocessing.Pool(processes=n_workers) as pool:
        while not es.stop():
            # 1. Ask for new solutions
            solutions = es.ask()
            
            # 2. Evaluate in parallel
            # Partial function for fixed VAE/MDNRNN file paths 
            eval_func = partial(evaluate_controller, 
                                vae_filepath=vae_filepath, 
                                mdn_rnn_filepath=mdn_rnn_filepath) 
 
            rewards = pool.map(eval_func, solutions) 
 
            # 3. CMA-ES wants to minimize cost => use negative reward 
            fitness = [-r for r in rewards] 
 
            # 4. Pass results back to CMA-ES 
            es.tell(solutions, fitness) 
 
            # 5. Print status 
            best_idx = np.argmin(fitness)  # or best_idx = np.argmax(rewards) 
            print(f"Best Reward this gen: {rewards[best_idx]:.2f}") 
 
        best_sol = es.result.xbest 
        return best_sol 
 
 
def main(): 
    # 1. Filepaths to your pretrained models 
    vae_filepath = "VAE_1.pth" 
    mdn_rnn_filepath = "mdn_rnn.pth" 
 
    # 2. Optimize Controller in parallel 
    best_controller_weights = optimize_controller(vae_filepath, mdn_rnn_filepath) 
    print("Controller Optimization Complete.") 
 
    # 3. Save best weights
    save_directory = "controller_weights"
    os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist
    save_path = os.path.join(save_directory, "best_controller_weights.npy")
    np.save(save_path, best_controller_weights)
    print(f"Best controller weights saved to {save_path}")

    # 4. Final test with best weights (single process)
    final_reward = evaluate_controller(best_controller_weights,
                                       vae_filepath,
                                       mdn_rnn_filepath)
    print(f"Final Test Reward: {final_reward:.2f}")


if __name__ == "__main__":
    main()
