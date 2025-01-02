import os
import cv2
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from VAE import VAE
from MDNRNN import MDNRNN

# === Same transforms used by controller.py ===
z_dim = 1024
hidden_dim = 256
action_dim = 3
mdn_temperature = 2.0

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
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
                           hidden_dim=hidden_dim, num_mixtures=5)
    mdn_rnn_model.load_state_dict(torch.load(filepath, map_location="cpu"))
    mdn_rnn_model.eval()  # Set model to evaluation mode
    return mdn_rnn_model

def preprocess_rgb_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Preprocess an RGB frame from the environment (H×W×3, uint8) into a (1, 3, 64, 64) FloatTensor.
    """
    frame_tensor = transform(frame)  # [3,64,64] in [0..1]
    return frame_tensor.unsqueeze(0)  # (1,3,64,64)

def main():
    # --------------------------------
    # 1) Load Environment in rgb_array
    # --------------------------------
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, info = env.reset()

    # Number of steps to run
    MAX_STEPS = 1000

    # --------------------------------
    # 2) Load VAE, MDN-RNN, and Controller
    # --------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = load_vae_model("VAE_1.pth").to(device)
    mdn_rnn = load_mdn_rnn_model("mdn_rnn.pth").to(device)

    # Load best controller weights (found by CMA-ES) 
    controller_weights = np.load("controller_weights/best_controller_weights.npy")
    # Reshape into a matrix: (action_dim × (z_dim + hidden_dim))
    weight_matrix = controller_weights.reshape(action_dim, z_dim + hidden_dim)

    # Hidden states for MDN-RNN
    h = None
    c = None

    # --------------------------------
    # 3) Set up Video Writer
    #    We'll do side-by-side: 64×64 + 64×64 => 128×64
    # --------------------------------
    video_filename = "dream_controller_side_by_side.mp4"
    fps = 30
    out_width = 64 * 2
    out_height = 64

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_filename, fourcc, fps, (out_width, out_height))

    # --------------------------------
    # 4) Run the Environment Loop
    # --------------------------------
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        step_count += 1

        # (A) Grab the latest rendered frame (RGB)
        frame_rgb = env.render()  # shape (H, W, 3), uint8

        # (B) Preprocess and encode with VAE
        frame_tensor = preprocess_rgb_frame(frame_rgb).to(device)   # shape: (1,3,64,64)
        with torch.no_grad():
            z_mu, z_logvar = vae.encoder(frame_tensor)
            # Optionally sample stochastically; or just use mu if you prefer deterministic
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mu + eps * std  # shape => (1, z_dim)

        # (C) Prepare hidden-state if None
        if h is None or c is None:
            h = torch.zeros(1, 1, hidden_dim, device=device)
            c = torch.zeros(1, 1, hidden_dim, device=device)

        # Flatten z and hidden state
        z_np = z.squeeze(0).cpu().numpy()  # shape (z_dim,)
        h_np = h.squeeze(0).squeeze(0).cpu().numpy()  # shape (hidden_dim,)

        # Combine them for the controller input
        controller_input = np.hstack([z_np, h_np])  # length = z_dim + hidden_dim

        # (D) Compute action from the controller
        action = weight_matrix @ controller_input  # shape => (3,)
        # Clip action to valid ranges
        action[0] = np.clip(action[0], -1., 1.)  # steering
        action[1] = np.clip(action[1],  0.,  1.) # gas
        action[2] = np.clip(action[2],  0.,  1.) # brake

        # (E) Step the environment 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # (F) Pass (z, action) to MDN-RNN to update hidden state
        z_seq = z.unsqueeze(1)  # shape (1,1,z_dim)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
        action_tensor = action_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,3)
        with torch.no_grad():
            _, _, _, (h, c) = mdn_rnn(z_seq, action_tensor, (h, c), temperature=mdn_temperature)

        # (G) Reconstruct the frame from z (for the side-by-side)
        with torch.no_grad():
            recon = vae.decoder(z)  # shape => (1,3,64,64)
            recon_np = recon.squeeze(0).permute(1,2,0).cpu().numpy()
            recon_np = (recon_np * 255).astype(np.uint8)  # [0..255]

        # (H) Resize original frame to 64×64
        pil_original = Image.fromarray(frame_rgb)
        pil_original = pil_original.resize((64, 64), Image.BICUBIC)
        orig_np = np.array(pil_original)  # shape => (64,64,3), uint8

        # Convert both images to BGR for OpenCV
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)

        # (I) Combine side-by-side
        side_by_side = np.hstack([orig_bgr, recon_bgr])
        out_video.write(side_by_side)

        if done:
            print(f"Episode finished at step={step_count} with reward={reward:.2f}")
            break

    # --------------------------------
    # 5) Clean Up
    # --------------------------------
    env.close()
    out_video.release()
    print(f"Side-by-side dream video saved to: {video_filename}")

if __name__ == "__main__":
    main()
