# JK2500-WorldModel

A repository showcasing a **World Model** approach for the [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) environment. It includes:

1. **Variational Autoencoder (VAE)** for image compression into latent representations.  
2. **Mixture Density Network + RNN (MDN-RNN)** for learning temporal dynamics in the latent space.  
3. **Controller** that uses the learned latent model to optimize policy actions (via CMA-ES or similar).  
4. **Training scripts for PPO** (separately, to generate data and also to demonstrate standard RL).  
5. **Scripts to collect environment frames & actions** into datasets.  
6. **Script (`Dream.py`)** to generate a "hallucinated" or "dreamed" rollout using only the VAE + MDN-RNN, producing new frames as a video.

---

## Repository Structure

```
jk2500-WorldModel/
├── Dream.py
├── controller.py
├── MDNRNN.py
├── VAE.py
├── CarRacing.py
├── CarRacingGym.py
├── CarRacingImagesActions.py
├── VAE_1.pth
└── mdn_rnn.pth
```

---

## Main Files

### CarRacing.py

- Trains a PPO agent (from Stable-Baselines3) on the CarRacing-v3 environment.
- Saves the best model and logs in `./car_racing_logs/`.

### CarRacingGym.py

- Loads the best PPO model trained in CarRacing.py and runs it interactively in a human-rendered mode.

### CarRacingImagesActions.py

- Uses the trained PPO model to collect episodes.
- Renders and saves the frames (.png images) and logs the actions (`actions.txt`) at specified intervals.
- The collected frames and actions can be used for training the VAE and MDN-RNN.

### VAE.py

- Contains implementation of a Variational Autoencoder for compressing 64×64 RGB images into a 1024-dimensional latent vector.
- The script includes Encoder, Decoder, and a VAE wrapper class.
- Also includes a (commented out) training loop for the VAE.

### MDNRNN.py

- Implementation of an MDN-RNN, which predicts the next latent vector given the current latent vector and action.
- The MDN output predicts mixture parameters (π, µ, σ).
- Includes a training script (`train_mdnrnn()`) that loads multi-episode data from a specified folder structure.

### controller.py

- Demonstrates how one might evolve or optimize a controller that uses the VAE + MDN-RNN’s internal state.
- Uses CMA-ES (cma library) to find action weights that maximize cumulative reward in a simulated loop with the MDN-RNN.

### Dream.py

- Loads pre-trained `VAE_1.pth` and `mdn_rnn.pth`.
- Takes the first frame of a recorded CarRacing episode, encodes it with the VAE, and then rolls forward in latent space via the MDN-RNN for 999 steps.
- Decodes each predicted latent back to an image and stitches them into a video (.mp4).

---

## Pre-trained Checkpoints

- **VAE_1.pth**: Pre-trained VAE weights.
- **mdn_rnn.pth**: Pre-trained MDN-RNN weights.

---

## Usage

Below is a general outline of how to use the code in this repository. Feel free to adapt based on your workflow or platform (e.g., local machine vs. cloud, CPU vs. GPU).

### 1. Install Dependencies

**Python 3.8+ recommended**

Install required packages (using pip, conda, or poetry, etc.):

```bash
pip install torch torchvision stable-baselines3 gymnasium matplotlib cma opencv-python
```

**Note:**

- For Apple Silicon (M1/M2), ensure you have the correct torch wheels installed (e.g., `torch>=2.0, <2.1` with mps support).
- If you need to record or edit videos, install FFmpeg.
- For GPU usage on NVIDIA, install CUDA-compatible PyTorch.

### 2. Train a PPO Policy (Optional)

If you want to train your own CarRacing agent from scratch, run:

```bash
python CarRacing.py
```

This will:

- Train PPO on CarRacing-v3 for 500k timesteps (configurable in code).
- Save logs and the best model in `./car_racing_logs/`.

### 3. Run the Trained Agent

To play the trained model (the best or final PPO checkpoint) in a local interactive window:

```bash
python CarRacingGym.py
```

This will load `car_racing_logs/best_model.zip` by default.

Press `Esc` to exit the CarRacing environment early if needed.

### 4. Collect Data (Frames & Actions)

The script `CarRacingImagesActions.py` runs the trained PPO agent to collect (RGB) frames and actions for further training the VAE and MDN-RNN.

By default, it saves data in `MDN-RNN/episode_*/`.

You can configure `NUM_EPISODES`, `EPISODE_LENGTH`, and `SAVE_INTERVAL` at the top of the script:

```bash
python CarRacingImagesActions.py
```

It will output images (`Img_step_XX.png`) and `actions.txt` (logging `[steering, gas, brake]` each step).

### 5. Train the VAE

Inside `VAE.py`, you’ll find a skeleton training loop (commented out).

Modify the dataset path and hyperparameters, then un-comment and run directly:

```python
# dataset = ImageFolder(root="data", transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# model = VAE(latent_dim=1024).to(device)
# ...
# for epoch in range(num_epochs):
#     ...
# torch.save(model.state_dict(), "VAE_1.pth")
```

Alternatively, if you already have `VAE_1.pth`, skip this step.

### 6. Train the MDN-RNN

`MDNRNN.py` includes a function `train_mdnrnn(...)`.

It looks for data in a structure like:

```
MDN-RNN/
  episode_1/
    representations_z.npy
    representations_mu.npy
    actions.txt
  ...
```

**Note:** You need to first transform raw images into latent vectors (`representations_z.npy` / `representations_mu.npy`). This is not explicitly shown in the code but typically done by running your VAE on each image to produce latents.

After you have the `.npy` latent files and `actions.txt`, run:

```bash
python MDNRNN.py
```

This will train an MDN-RNN and save weights to `mdn_rnn.pth`.

### 7. Generate Dream Rollouts (Dream.py)

Once you have:

- A trained VAE (`VAE_1.pth`)
- A trained MDN-RNN (`mdn_rnn.pth`)
- A set of recorded actions (in some `actions.txt`)

Adjust the paths near the top of `Dream.py` (e.g., `actions_file` and `frames_folder`), then run:

```bash
python Dream.py
```

**What it does:**

- Loads the first frame from an episode.
- Encodes it to latent `z_0`.
- Iteratively uses the MDN-RNN to predict the next latent `z_{t+1}` given (`z_t`, `a_t`).
- Decodes each predicted latent frame via the VAE decoder.
- Writes frames to a `.mp4` video (`predicted_frames_1_to_999.mp4`).

### 8. Controller Optimization via CMA-ES (Optional)

`controller.py` shows an experimental method to learn direct action weights from the hidden state (`h, c`) and latent `z_t` using CMA-ES.

It does not rely on the PPO approach; it aims to find an action policy purely in latent space.

To run:

```bash
python controller.py
```

**The script:**

- Loads `VAE_1.pth` and `mdn_rnn.pth`.
- Creates a CarRacing environment.
- Evolves a linear controller (mapping `[z, h] -> actions`) via CMA-ES to maximize reward.
- Prints the best controller parameters and final test reward.

---

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium (and Box2D dependencies if needed)
- Matplotlib (for saving images)
- OpenCV (for video writing)
- cma (for controller optimization)

---

## Notes & Tips

### Apple Silicon (M1/M2) Support

- Much of the code references `mps` device. If you’re on macOS with an M1 or M2 chip, make sure you install a version of PyTorch that supports Metal Performance Shaders (MPS).

### Data Collection

- The repository scripts rely on a simple flow: first collect images & actions (via a trained PPO), then process them offline to generate latent vectors for the MDN-RNN training.

### Video Rendering

- `Dream.py` uses OpenCV to write `.mp4` files. If you have any issues with codecs, install FFmpeg or check that your OpenCV was compiled with ffmpeg support.

### Hyperparameters

- The scripts generally use large latent dimensions (1024). You can experiment with smaller `z_dim` to speed up training.

### Environment Differences

- The environment `CarRacing-v3` has slightly different observation shapes or reward structure than previous versions. Make sure to install a Gymnasium version that supports it.

---

## Acknowledgments

- David Ha and Jürgen Schmidhuber’s World Models for the inspiration behind training a VAE + MDN-RNN for environment modeling.
- Stable-Baselines3 for the PPO RL implementation.
- The open-source community for continued improvements to RL, Gym, and PyTorch libraries.

---

## License

This project is provided under an open-source license. See `LICENSE` for details (if you have a LICENSE file). Otherwise, adapt or include whichever license you prefer.

