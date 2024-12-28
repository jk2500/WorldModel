import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from VAE import VAE
from MDNRNN import MDNRNN
import os
from PIL import Image
import pandas as pd
import numpy as np

# -----------------------------------------------------
# 1) Load Pre-trained Models
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Instantiate your VAE and load weights
vae = VAE(latent_dim=1024).to(device)
vae.load_state_dict(torch.load("VAE_1.pth", map_location=device))
vae.eval()

# Instantiate your MDN-RNN and load weights
mdn_rnn = MDNRNN(z_dim=1024, action_dim=3, hidden_dim=256, num_mixtures=5).to(device)
mdn_rnn.load_state_dict(torch.load("mdn_rnn.pth", map_location=device))
mdn_rnn.eval()

# -----------------------------------------------------
# 2) Prepare Input Data
#    - Single frame_0
#    - actions: tensor of shape (1000, action_dim)
# -----------------------------------------------------
actions_file = "MDN-RNN/episode_1/actions.txt"  # Replace with the actual file path
actions_df = pd.read_csv(actions_file)

# Convert the string representations of arrays into actual arrays
actions_list = [np.fromstring(action.strip("[]"), sep=" ") for action in actions_df["Action"]]
actions = torch.tensor(actions_list, dtype=torch.float32).to(device)

# Define the folder containing the images
frames_folder = "MDN-RNN/episode_1"  # Replace with the actual folder path

# Image preprocessing
image_size = (64, 64)  # Resize to match VAE input size
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()  # Convert to tensor and normalize to [0, 1]
])

# Load only the first frame
frame_0_path = os.path.join(frames_folder, "Img_step_50.png")  # Assuming .png format
img = Image.open(frame_0_path).convert("RGB")
frame_0 = preprocess(img).unsqueeze(0).to(device)  # shape: (1, 3, 64, 64)

# -----------------------------------------------------
# 3) Encode the first frame with VAE to get latent z_0
# -----------------------------------------------------
with torch.no_grad():
    mu, logvar = vae.encoder(frame_0)
    # Deterministic choice: use mu
    z_0 = mu  # shape => (1, 1024)

# -----------------------------------------------------
# 4) "Teacher force" the first step into the MDN-RNN
#    so we get the final hidden state (h, c) at t=0
# -----------------------------------------------------
# We'll create a single-step (batch=1, time=1) input
z_0_1 = z_0.unsqueeze(1)        # shape => (1, 1, 1024)
a_0_1 = actions[0].unsqueeze(0) # shape => (1, 3)
a_0_1 = a_0_1.unsqueeze(1)      # shape => (1, 1, 3)

with torch.no_grad():
    pi, mu, sigma, (h, c) = mdn_rnn(z_0_1, a_0_1)
    # h, c => each of shape (1, 1, hidden_dim)

# Now z_current is the latent z_0
z_current = z_0  # shape: (1, 1024)

# -----------------------------------------------------
# 5) Predict frames t=1..999 (999 frames total)
#    We'll do a closed-loop rollout:
#    z_t -> (z_t, a_t) -> MDN -> z_{t+1}
# -----------------------------------------------------
predicted_latents = []
with torch.no_grad():
    for t in range(51, 1000):
        z_input = z_current.unsqueeze(1)  # shape => (1, 1, 1024)
        a_input = actions[t].unsqueeze(0).unsqueeze(1)  # shape => (1, 1, 3)

        pi_t, mu_t, sigma_t, (h, c) = mdn_rnn(z_input, a_input, (h, c))
        # pi_t:   (1, 1, K)
        # mu_t:   (1, 1, K, z_dim)
        # sigma_t:(1, 1, K, z_dim)

        # Pick the most likely mixture component
        pi_argmax = torch.argmax(pi_t[:, 0, :], dim=-1)  # shape (1,)
        next_z = mu_t[:, 0, pi_argmax, :].squeeze(1)     # shape => (1, 1024)

        predicted_latents.append(next_z.squeeze(0).cpu())  # shape => (1024,)
        z_current = next_z

predicted_latents = torch.stack(predicted_latents, dim=0)  # shape => (999, 1024)

# -----------------------------------------------------
# 6) Decode predicted latents into frames
# -----------------------------------------------------
generated_frames = []
vae.eval()
with torch.no_grad():
    for i in range(predicted_latents.shape[0]):
        z_i = predicted_latents[i].unsqueeze(0).to(device)  # (1, 1024)
        x_recon = vae.decoder(z_i)                          # shape => (1, 3, 64, 64)

        # Convert to CPU numpy image in [0,1], shape => (64,64,3)
        frame_np = x_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_frames.append(frame_np)

# -----------------------------------------------------
# 7) Write the predicted frames t=1..999 to a video
# -----------------------------------------------------
output_video_path = "predicted_frames_1_to_999.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30
height, width = generated_frames[0].shape[:2]
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for frame_rgb in generated_frames:
    # Scale [0,1] -> [0,255]
    frame_bgr = (frame_rgb * 255).astype('uint8')
    # Convert RGB -> BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    out_writer.write(frame_bgr)

out_writer.release()
print(f"Predicted video (frames 1..999) saved to: {output_video_path}")
