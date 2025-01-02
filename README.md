# JK2500-WorldModel

A repository showcasing a **World Model** approach for the [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) environment. This approach is inspired by [David Ha and Jürgen Schmidhuber’s "World Models"](https://worldmodels.github.io/) paper. The main idea is to learn a compressed representation of the environment through a **Variational Autoencoder (VAE)**, model its temporal dynamics with an **MDN-RNN**, and then use these learned components for policy optimization (via CMA-ES or a standard RL algorithm like PPO).

## Contents

1. **Variational Autoencoder (VAE)** — compresses 64×64 RGB frames into a 1024-dimensional latent space.  
2. **Mixture Density Network + RNN (MDN-RNN)** — learns temporal dynamics in latent space.  
3. **Controller** — demonstrates how one might optimize actions based on the latent state and MDN-RNN hidden states, using CMA-ES.  
4. **PPO Training Scripts** — trains a CarRacing agent (to collect data or as a baseline).  
5. **Data Collection Scripts** — captures (RGB) frames and corresponding actions for training the VAE & MDN-RNN.  
6. **Dream Rollouts** (`Dream.py`) — uses only the VAE + MDN-RNN to produce a "hallucinated" rollout as a video.

---

## Repository Structure

```
jk2500-WorldModel/
├── CarRacing.py               # Trains PPO agent on CarRacing-v3
├── CarRacingGym.py            # Loads and runs the PPO agent interactively
├── CarRacingImagesActions.py  # Collects frames & actions from a trained PPO
├── controller.py              # CMA-ES based controller in latent space
├── Dream.py                   # Generates hallucinated rollouts using VAE + MDN-RNN
├── MDNRNN.py                  # Implementation + training code for the MDN-RNN
├── VAE.py                     # Implementation of the VAE (encoder + decoder + training logic)
├── VAE_1.pth                  # Pre-trained VAE weights (example)
├── mdn_rnn.pth                # Pre-trained MDN-RNN weights (example)
└── README.md                  # Project documentation (this file)
```

---

## Overview of Main Files

### 1. `CarRacing.py`
- **Purpose**: Train a PPO agent (via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)) on CarRacing-v3.  
- **Details**:
  - Uses a CNN policy to handle image-based observations.
  - Saves tensorboard logs, checkpoint models, and `best_model.zip` in `./car_racing_logs/`.
  - Contains an `EvalCallback` that can stop training early upon reaching a reward threshold.

### 2. `CarRacingGym.py`
- **Purpose**: Loads the best or final PPO checkpoint and runs it in a local, human-rendered window for evaluation or demonstration.  
- **Usage**:
  - By default, attempts to load `car_racing_logs/best_model.zip`.
  - Press `Esc` to exit the environment early.

### 3. `CarRacingImagesActions.py`
- **Purpose**: Collect a dataset of frames and actions from a trained PPO agent.  
- **Details**:
  - Runs the agent in `rgb_array` mode, capturing frames (`.png`) every `SAVE_INTERVAL` steps and logging actions to `actions.txt`.
  - Organizes data into `MDN-RNN/episode_X/` directories (where X is the episode number).
  - This data can be used offline to train the VAE and MDN-RNN.

### 4. `VAE.py`
- **Purpose**: Implementation of the Variational Autoencoder for 64×64 RGB frames.  
- **Key Components**:
  - **Encoder**: 4 convolutional layers reducing a 64×64×3 image into a 1024-D vector (mu and logvar).  
  - **Decoder**: Uses transposed convolutions to reconstruct 64×64 images from a 1024-D latent.  
  - **Training**: Includes a sample training loop (commented in `main()`), with a `vae_loss` function combining reconstruction (BCE) and KL divergence.

### 5. `MDNRNN.py`
- **Purpose**: Implementation and training routine for a Mixture Density Network RNN.  
- **Details**:
  - **Input**: (z, action) at each timestep, where `z` is the latent from the VAE.  
  - **Output**: Parameters of a mixture of Gaussians (π, µ, σ) predicting the next latent vector `z_{t+1}`.  
  - **MDN Loss**: Negative log-likelihood of the true latent under the predicted mixture distribution.  
  - **Training**:
    - A function `train_mdnrnn(...)` that looks for data in folders like `MDN-RNN/episode_*`.
    - Each episode folder should contain `representations_z.npy`, `representations_mu.npy`, and `actions.txt`.

### 6. `controller.py`
- **Purpose**: Demonstrates how one might optimize a policy (controller) entirely in latent space using CMA-ES (rather than PPO).  
- **Approach**:
  - Loads `VAE_1.pth` and `mdn_rnn.pth`.
  - A linear mapping from `[z, h] -> actions` is parameterized; CMA-ES searches for the best parameters to maximize reward in CarRacing.  
  - Each candidate is evaluated by stepping in the real environment but using the MDN-RNN to predict future latent states.

### 7. `Dream.py`
- **Purpose**: Generate "hallucinated" or "dreamed" rollouts by using only the VAE + MDN-RNN (no real environment).  
- **How It Works**:
  - Loads an initial frame from a real episode (or a manually selected frame).
  - Encodes it to get `z_0`.
  - Iteratively predicts next latent `z_{t+1}` using MDN-RNN and decodes it back to an image.  
  - Produces a video of predicted frames (`.mp4`).

---

## Pre-trained Checkpoints

- **`VAE_1.pth`**: Example of a pre-trained VAE.  
- **`mdn_rnn.pth`**: Example of a pre-trained MDN-RNN.  

If these are not provided or you want to train your own, see the training instructions below.

---

## Getting Started

Below is a quick-start guide. Adjust paths and hyperparameters as needed.

### 1. Install Dependencies

**Python 3.8+ recommended**  
Install the required packages (this example uses `pip`):

```bash
pip install torch torchvision stable-baselines3 gymnasium matplotlib cma opencv-python
```

**Additional Notes**:

- For Apple Silicon (M1/M2), ensure a compatible PyTorch build with MPS support (e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu` or as recommended by PyTorch docs).  
- If you need video rendering (`.mp4`) support, install [FFmpeg](https://ffmpeg.org/).  
- For NVIDIA GPUs, install CUDA-compatible PyTorch wheels.

### 2. Train a PPO Policy (Optional)

Run:

```bash
python CarRacing.py
```

- Trains PPO for ~500k timesteps on CarRacing-v3 (you can adjust `training_timesteps`).  
- Model checkpoints and logs go to `./car_racing_logs/`.

### 3. Run the Trained PPO Agent

To see your trained agent in action:

```bash
python CarRacingGym.py
```

- Loads `car_racing_logs/best_model.zip` by default (or final checkpoint if `best_model.zip` isn’t available).  
- Renders the environment in a window.

### 4. Collect Data for VAE & MDN-RNN

Run:

```bash
python CarRacingImagesActions.py
```

- Captures environment frames (.png) and actions in `MDN-RNN/episode_*`.  
- Configure `NUM_EPISODES`, `EPISODE_LENGTH`, and `SAVE_INTERVAL` inside the script.

### 5. Train the VAE

Inside `VAE.py`:

- There is a sample training loop (commented out in `main()`) which uses an `ImageFolder` dataset.  
- You can place your collected PNG frames into a suitable folder structure for `ImageFolder`. Or adapt the loading logic to your data.  
- Run (after un-commenting the training code in `main()`):

```bash
python VAE.py
```

- Saves `VAE_1.pth` upon completion.

### 6. Train the MDN-RNN

```bash
python MDNRNN.py
```

- Looks for directories `MDN-RNN/episode_*/` containing `representations_z.npy`, `representations_mu.npy`, and `actions.txt`.  
- In practice, you need to run your VAE’s encoder on each saved frame to generate those `.npy` files. (Scripts for that step are not included here but are straightforward to implement using `VAE.encoder(...)`.)

Once done, you get `mdn_rnn.pth`.

### 7. Generate a "Dream" Rollout

```bash
python Dream.py
```

- Uses the loaded `VAE_1.pth` and `mdn_rnn.pth` to produce a side-by-side video: left = original frame, right = reconstruction from the VAE.  
- Steps forward in time purely in latent space (via MDN-RNN).  

### 8. Latent-Space Controller Optimization (Optional)

```bash
python controller.py
```

- Evolves a linear controller from `[z_t, hidden_state] -> [steering, gas, brake]` using [CMA-ES](https://github.com/CMA-ES/pycma).  
- Saves the best weights to `controller_weights/best_controller_weights.npy`.  
- Prints the best test reward.

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- Stable-Baselines3
- Gymnasium (with Box2D dependencies)
- Matplotlib
- OpenCV (for video writing)
- cma

---

## Notes & Tips

1. **Apple Silicon (M1/M2)**:  
   - If running on MPS, check code references to `device="mps"` in `CarRacing.py` or `VAE.py`.  
   - Install the correct PyTorch version for MPS acceleration.

2. **Collecting Data**:  
   - Typical flow is to train a PPO agent first, then run `CarRacingImagesActions.py` to gather frames and actions for VAE/MDN-RNN training.  
   - Alternatively, any policy (random or hand-coded) can generate data.

3. **Hyperparameters**:  
   - Current defaults use a 1024-D latent (`z_dim=1024`) which can be expensive. Consider smaller latent sizes (e.g., 64–256) if you need faster training.

4. **Video Rendering**:  
   - `Dream.py` uses OpenCV to write an `.mp4`. If you encounter codec issues, ensure you have FFmpeg installed or modify the `fourcc` to a suitable codec.

5. **MDN-RNN**:  
   - The default number of mixtures is 5. Higher mixtures can capture more complex distributions but also increase training time.

6. **Controller vs. PPO**:  
   - The `controller.py` script is an alternative approach to learning actions. It bypasses typical RL policies (like PPO) in favor of a direct search in the parameter space of a linear policy.

---

## Acknowledgments

- Inspired by [David Ha and Jürgen Schmidhuber’s "World Models"](https://worldmodels.github.io/).  
- Thanks to the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) team and open-source contributors for excellent RL frameworks.

---

## License


```
MIT License
Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
...
