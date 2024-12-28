# train_mdnrnn.py
import os
import glob
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# =================================
#     1) Model Components
# =================================


class MDNOutput(nn.Module):
    """
    Mixture Density Network (MDN) output layer.
    Given an LSTM hidden state, predicts parameters of a mixture of diagonal Gaussians:
      - pi (mixture weights)
      - mu (means)
      - sigma (std dev)
    With optional temperature scaling applied during forward().
    """
    def __init__(self, hidden_dim, z_dim, num_mixtures=5):
        super(MDNOutput, self).__init__()
        self.z_dim = z_dim
        self.num_mixtures = num_mixtures

        # For each mixture:
        #   - z_dim for mu
        #   - z_dim for sigma
        #   - 1 for pi
        # So total = (2*z_dim + 1) per mixture
        self.param_dim = (2 * z_dim + 1) * num_mixtures

        self.fc = nn.Linear(hidden_dim, self.param_dim)

    def forward(self, h, temperature=1.0):
        """
        h: (batch_size, seq_len, hidden_dim)
        temperature: float

        Returns:
          pi -> (batch_size, seq_len, num_mixtures)
          mu -> (batch_size, seq_len, num_mixtures, z_dim)
          sigma -> (batch_size, seq_len, num_mixtures, z_dim)
        """
        # shape of out: (B, T, param_dim)
        out = self.fc(h)

        # reshape to separate mixture components
        # (B, T, K, 2*z_dim + 1)
        out = out.view(out.size(0), out.size(1), self.num_mixtures, 2*self.z_dim + 1)

        # Slice out pi, mu, sigma
        pi = out[:, :, :, 0]                             # (B, T, K)
        mu = out[:, :, :, 1:1+self.z_dim]                # (B, T, K, z_dim)
        sigma = out[:, :, :, 1+self.z_dim:1+2*self.z_dim]# (B, T, K, z_dim)

        # -------- Apply temperature scaling --------
        # 1) Scale pi logits by (1 / temperature) before softmax
        pi = F.softmax(pi / temperature, dim=-1)

        # 2) Exponentiate sigma, then multiply by sqrt(T) (or T)
        sigma = torch.exp(sigma) * math.sqrt(temperature)
        # -------------------------------------------

        return pi, mu, sigma
    
class MDNRNN(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim=256, num_mixtures=5):
        super(MDNRNN, self).__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures

        # LSTM: input_size = z_dim + action_dim
        self.lstm = nn.LSTM(input_size=z_dim + action_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)

        # The MDN output head (with temperature support)
        self.mdn = MDNOutput(hidden_dim, z_dim, num_mixtures=num_mixtures)

    def forward(self, z_seq, a_seq, h_c=None, temperature=1.0):
        """
        z_seq: (B, T, z_dim)
        a_seq: (B, T, action_dim)
        h_c: optional (h0, c0) for LSTM -> each (1, B, hidden_dim)
        temperature: float (controls MDN sharpness in pi and sigma)

        Returns:
          pi, mu, sigma, (h, c)
        """
        # Concatenate (z, a) along the features dimension
        inp = torch.cat([z_seq, a_seq], dim=-1)  # (B, T, z_dim+action_dim)
        out, (h, c) = self.lstm(inp, h_c)        # out: (B, T, hidden_dim)

        pi, mu, sigma = self.mdn(out, temperature=temperature)
        return pi, mu, sigma, (h, c)

# =====================================================
# 2) MDN Loss Function (Negative Log Likelihood)
# =====================================================
def mdn_loss_fn(z_true, pi, mu, sigma):
    """
    z_true: (B, T, z_dim)
    pi:     (B, T, K)
    mu:     (B, T, K, z_dim)
    sigma:  (B, T, K, z_dim)

    returns scalar negative log-likelihood
    """
    # Expand z_true to (B, T, 1, z_dim) so it can broadcast vs. (B, T, K, z_dim)
    z_true_expanded = z_true.unsqueeze(2)  # (B, T, 1, z_dim)

    # log probability under each Gaussian component:
    #   log_gauss_k = -0.5 * ((z - mu)/sigma)^2 - sum(log(sigma)) - (z_dim/2)*log(2*pi)
    # We'll handle it carefully in log-space.

    var = sigma**2
    log_exponent = -0.5 * ((z_true_expanded - mu)**2) / (var + 1e-8)
    # sum across z_dim
    log_exponent = log_exponent.sum(dim=-1)  # (B, T, K)

    # log_det = sum(log(sigma)) across z_dim
    log_det = torch.log(sigma + 1e-8).sum(dim=-1)  # (B, T, K)

    log_const = z_true.size(-1) * 0.5 * math.log(2*math.pi)
    log_gauss = log_exponent - (log_det + log_const)

    # combine with mixture weights
    log_pi = torch.log(pi + 1e-8)  # (B, T, K)
    # log-sum-exp over mixture components
    log_probs = torch.logsumexp(log_pi + log_gauss, dim=-1)  # (B, T)

    # negative log-likelihood
    nll = -log_probs.mean()  # average over batch & time
    return nll




# =======================================
# 3) MultiEpisodeDataset
# =======================================
class MultiEpisodeDataset(Dataset):
    """
    Loads multiple episodes from a directory structure, e.g.:
        data/
          episode_000/
            z.npy
            mu.npy
            actions.csv   # lines like "Step,Action"
          episode_001/
            z.npy
            mu.npy
            actions.csv
          ...
    Returns one entire episode at a time: (z_seq, mu_seq, actions_seq).
    """
    def __init__(self, root_dir):
        super().__init__()
        # find all subfolders named 'episode_*'
        self.episode_dirs = sorted(glob.glob(os.path.join(root_dir, "episode_*")))

        self.episodes = []
        for e_dir in self.episode_dirs:
            z_path = os.path.join(e_dir, "representations_z.npy")
            mu_path = os.path.join(e_dir, "representations_mu.npy")
            actions_path = os.path.join(e_dir, "actions.txt")  # or however it's named

            # Load z and mu
            z_seq = np.load(z_path)   # shape: (T, z_dim)
            mu_seq = np.load(mu_path) # shape: (T, z_dim)

            # Read and parse actions.csv
            with open(actions_path, "r") as f:
                lines = f.read().strip().split("\n")
            # The first line is header: "Step,Action"
            # Subsequent lines are like: "0,[0.84940237 1.         0.        ]"
            lines = lines[1:]  # skip header line
            actions = []
            for line in lines:
                # e.g. "0,[0.84940237 1.         0.        ]"
                step_str, arr_str = line.split(",", 1)
                # remove brackets
                arr_str = arr_str.strip().strip("[]")
                # split by whitespace
                float_strs = arr_str.split()
                action = [float(x) for x in float_strs]
                actions.append(action)

            actions = np.array(actions, dtype=np.float32)  # shape: (T, action_dim)

            # Confirm shapes match
            T = z_seq.shape[0]
            assert actions.shape[0] == T, f"Mismatch in T for {e_dir}"
            assert mu_seq.shape[0] == T, f"Mismatch in T for {e_dir}"

            # Convert z_seq, mu_seq to float32 if not already
            z_seq = z_seq.astype(np.float32)
            mu_seq = mu_seq.astype(np.float32)

            self.episodes.append((z_seq, mu_seq, actions))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]  # (z_seq, mu_seq, actions)


def train_mdnrnn(root_dir,
                 z_dim=16,
                 action_dim=3,
                 hidden_dim=256,
                 num_mixtures=5,
                 batch_size=1,
                 epochs=10,
                 lr=1e-3,
                 save_model_path="mdn_rnn.pth"):
    dataset = MultiEpisodeDataset(root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MDNRNN(z_dim=z_dim,
                   action_dim=action_dim,
                   hidden_dim=hidden_dim,
                   num_mixtures=num_mixtures)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for z_seq, mu_seq, act_seq in loader:
            optimizer.zero_grad()
            pi, mu, sigma, _ = model(z_seq, act_seq, temperature=1.0)
            loss = mdn_loss_fn(mu_seq, pi, mu, sigma)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        # Optionally save after each epoch
        # torch.save(model.state_dict(), save_model_path)
        # print(f"Saved model weights at epoch {epoch} -> {save_model_path}")

    # Or save once after all epochs:
    torch.save(model.state_dict(), save_model_path)
    print(f"Model weights saved to {save_model_path}")

    return model



# =======================================
# 5) Main Script Usage
# =======================================
if __name__ == "__main__":
    # Adjust root_dir if your data is stored elsewhere
    root_data_dir = "MDN-RNN"

    # Train the MDN-RNN
    trained_model = train_mdnrnn(
        root_dir=root_data_dir,
        z_dim=1024,
        action_dim=3,
        hidden_dim=256,
        num_mixtures=5,
        batch_size=2,  # adapt based on how many episodes you want per batch
        epochs=20,
        lr=1e-3
    )
    
    print("Training complete!")


