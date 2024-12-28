import torch
import numpy as np
from torchvision import transforms as T
from VAE import VAE
from MDNRNN import MDNRNN
import gymnasium as gym
from cma import CMAEvolutionStrategy

# === Set Hyperparameters ===
z_dim = 1024
hidden_dim = 256
action_dim = 3
num_mixtures = 5
controller_population_size = 4
mdn_temperature = 1.0

# Transformation for gym environment RGB output
transform = T.Compose([
    T.ToPILImage(),       # convert np.ndarray to PIL Image
    T.Resize((64, 64)),   # resize to 64x64
    T.ToTensor()          # convert PIL Image to torch.Tensor
])


def load_vae_model(filepath):
    vae_model = VAE(latent_dim=z_dim)
    vae_model.load_state_dict(torch.load(filepath))
    vae_model.eval()
    return vae_model

def load_mdn_rnn_model(filepath):
    mdn_rnn_model = MDNRNN(z_dim=z_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_mixtures=num_mixtures)
    mdn_rnn_model.load_state_dict(torch.load(filepath))
    mdn_rnn_model.eval()
    return mdn_rnn_model


def preprocess_rgb_frame(frame):
    """
    frame is a NumPy array of shape (height, width, 3) with uint8 values [0..255].
    This function will return a torch.Tensor of shape (1, 3, 64, 64) in [0..1].
    """
    # Just pass the NumPy array directly to the transform
    frame_tensor = transform(frame)     # shape (3,64,64)
    return frame_tensor.unsqueeze(0)    # shape (1,3,64,64)


def evaluate_controller(controller_weights, env, vae, mdn_rnn):
    cumulative_reward = 0
    obs, _ = env.reset()
    h, c = None, None

    while True:
        frame = env.render()
        frame_preprocessed = preprocess_rgb_frame(frame)
        z, _ = vae.encoder(frame_preprocessed)

        # Detach and convert to NumPy
        z_np = z.detach().numpy()
        h_np = h.detach().numpy() if h is not None else np.zeros(hidden_dim)
        
        action = np.dot(controller_weights, np.hstack([z_np, h_np]))
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Forward pass through MDN-RNN
        _, _, _, (h, c) = mdn_rnn.forward(z, action, (h, c), temperature=mdn_temperature)
        cumulative_reward += reward
        
        if terminated or truncated:
            break
    return cumulative_reward


def optimize_controller(env, vae, mdn_rnn, population_size=controller_population_size):
    num_params = z_dim + hidden_dim
    es = CMAEvolutionStrategy(np.zeros(num_params), 0.5, {"popsize": population_size})

    while not es.stop():
        solutions = es.ask()
        fitness = []

        for weights in solutions:
            reward = evaluate_controller(weights, env, vae, mdn_rnn)
            fitness.append(-reward)

        es.tell(solutions, fitness)
        print(f"Best Reward: {-min(fitness)}")
    return es.result.xbest

def main():
    # 1. Initialize Environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # 2. Load Pretrained Models
    vae = load_vae_model("VAE_1.pth")
    mdn_rnn = load_mdn_rnn_model("mdn_rnn.pth")

    # 3. Optimize Controller
    best_controller_weights = optimize_controller(env, vae, mdn_rnn)
    print("Controller Optimization Complete.")

    # 4. Test Best Controller
    reward = evaluate_controller(best_controller_weights, env, vae, mdn_rnn)
    print(f"Final Test Reward: {reward}")
if __name__ == "__main__":
    main()
