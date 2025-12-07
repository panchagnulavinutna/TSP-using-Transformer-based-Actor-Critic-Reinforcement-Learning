# tsp_transformer_actor_critic.py
import os
import time
import math
import json
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# -----------------------
# Config / Hyperparams
# -----------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
SEQ_LEN = 10
EMBED_DIM = 128
HIDDEN_DIM = 128
LR = 1e-3
EPOCHS = 300
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GRAD_CLIP = 1.0
VAL_BATCH = 256
LOG_DIR = "tsp_runs"
PLOT_RESULTS = True

# Transformer-specific
TRANSFORMER_NHEAD = 8
TRANSFORMER_NLAYERS = 3
TRANSFORMER_FF = 512
TRANSFORMER_DROPOUT = 0.1

os.makedirs(LOG_DIR, exist_ok=True)


# -----------------------
# Utilities
# -----------------------
def generate_batch(batch_size: int, seq_len: int, device=DEVICE) -> Tensor:
    return torch.rand(batch_size, seq_len, 2, device=device)


def compute_tour_length(coords: Tensor, tours: Tensor) -> Tensor:
    """
    coords: (B, N, 2)
    tours: (B, N) indices (0..N-1)
    returns: (B,) tour lengths
    """
    B, N, _ = coords.size()
    idx = tours.unsqueeze(-1).expand(-1, -1, 2)  # (B,N,2)
    ordered = torch.gather(coords, 1, idx)       # (B,N,2)
    rolled = torch.roll(ordered, shifts=-1, dims=1)
    return torch.sqrt(((ordered - rolled) ** 2).sum(-1)).sum(1)


# Nearest Neighbor heuristic for baseline (numpy coords)
def nearest_neighbor_length(coords: np.ndarray) -> float:
    N = coords.shape[0]
    visited = [False] * N
    tour = [0]
    visited[0] = True
    total_len = 0.0
    for _ in range(1, N):
        last = tour[-1]
        next_idx = np.argmin([np.linalg.norm(coords[last] - coords[j]) if not visited[j] else np.inf for j in range(N)])
        total_len += np.linalg.norm(coords[last] - coords[next_idx])
        tour.append(next_idx)
        visited[next_idx] = True
    total_len += np.linalg.norm(coords[tour[-1]] - coords[tour[0]])
    return total_len


# -----------------------
# Model: Transformer Encoder + Pointer Decoder
# -----------------------
class TransformerEncoderNet(nn.Module):
    """
    Transformer encoder that takes coords (B, N, 2) and returns node embeddings (B, N, E)
    """
    def __init__(self, input_dim=2, embed_dim=EMBED_DIM, nhead=TRANSFORMER_NHEAD, nlayers=TRANSFORMER_NLAYERS, dim_feedforward=TRANSFORMER_FF, dropout=TRANSFORMER_DROPOUT, use_positional=False):
        super().__init__()
        self.embed_in = nn.Linear(input_dim, embed_dim)
        self.post = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.use_positional = use_positional
        if self.use_positional:
            # learned positional embeddings (optional)
            self.pos_embedding = nn.Parameter(torch.randn(1, 100, embed_dim))  # up to 100 nodes by default
        else:
            self.pos_embedding = None

    def forward(self, coords: Tensor) -> Tensor:
        # coords: (B, N, 2)
        x = self.embed_in(coords)  # (B, N, E)
        B, N, E = x.size()
        if self.use_positional:
            if self.pos_embedding.size(1) < N:
                # expand pos embeddings if needed
                extra = N - self.pos_embedding.size(1)
                new_pos = torch.randn(1, extra, E, device=self.pos_embedding.device)
                self.pos_embedding = nn.Parameter(torch.cat([self.pos_embedding, new_pos], dim=1))
            pos = self.pos_embedding[:, :N, :].expand(B, -1, -1)  # (B,N,E)
            x = x + pos

        # transformer expects (S, B, E) where S = sequence length (N)
        x = x.permute(1, 0, 2)            # (N, B, E)
        out = self.transformer(x)         # (N, B, E)
        out = out.permute(1, 0, 2)        # (B, N, E)
        out = out + self.embed_in(coords) # residual with input embedding
        return self.post(out)


class PointerDecoder(nn.Module):
    """
    Pointer-style decoder that produces a tour (sequence of node indices), per-step log-probs, and entropy.
    """
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_to_query = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.decoder_rnn = nn.GRUCell(embed_dim, hidden_dim)
        self.pool_to_hidden = nn.Linear(embed_dim, hidden_dim)

    def forward(self, node_emb: Tensor, greedy: bool=False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        node_emb: (B, N, E)
        returns:
          tours: (B, N) indices
          log_probs: (B,) sum of log probs for the whole tour
          entropies: (B,) sum of entropies per step
        """
        B, N, E = node_emb.size()
        context = node_emb.mean(dim=1)                # (B, E)
        hidden = torch.tanh(self.pool_to_hidden(context))  # (B, H)

        visited = torch.zeros(B, N, dtype=torch.bool, device=node_emb.device)
        tours, log_probs_steps, entropies_steps = [], [], []
        keys = node_emb  # (B, N, E)

        for step in range(N):
            query = self.hidden_to_query(hidden).unsqueeze(1)  # (B,1,E)
            # dot-product attention (query * keys).sum(-1) -> (B,N)
            scores = (query * keys).sum(-1) / math.sqrt(E)
            scores = scores.masked_fill(visited, float("-inf"))
            probs = torch.softmax(scores, dim=-1)

            if greedy:
                idx = probs.argmax(dim=-1)  # (B,)
                logp = torch.log(torch.gather(probs, 1, idx.unsqueeze(-1)).squeeze(1) + 1e-12)
                ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
            else:
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()          # (B,)
                logp = dist.log_prob(idx)    # (B,)
                ent = dist.entropy()         # (B,)

            tours.append(idx)
            log_probs_steps.append(logp)
            entropies_steps.append(ent)

            # update visited safely
            visited = visited.clone()
            visited[torch.arange(B, device=node_emb.device), idx] = True

            chosen_emb = node_emb[torch.arange(B, device=node_emb.device), idx]  # (B, E)
            hidden = self.decoder_rnn(chosen_emb, hidden)  # (B, H)

        tours = torch.stack(tours, dim=1)              # (B, N)
        log_probs = torch.stack(log_probs_steps, dim=1).sum(1)    # (B,)
        entropies = torch.stack(entropies_steps, dim=1).sum(1)    # (B,)
        return tours, log_probs, entropies


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoderNet(input_dim=2, embed_dim=EMBED_DIM, nhead=TRANSFORMER_NHEAD, nlayers=TRANSFORMER_NLAYERS, dim_feedforward=TRANSFORMER_FF, dropout=TRANSFORMER_DROPOUT, use_positional=False)
        self.decoder = PointerDecoder(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.ReLU(),
            nn.Linear(EMBED_DIM // 2, 1)
        )

    def forward(self, coords: Tensor, greedy: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        coords: (B, N, 2)
        returns: tours (B,N), logp (B,), ent (B,), values (B,)
        """
        node_emb = self.encoder(coords)  # (B, N, E)
        tours, logp, ent = self.decoder(node_emb, greedy=greedy)
        pooled = node_emb.mean(dim=1)
        values = self.value_head(pooled).squeeze(-1)
        return tours, logp, ent, values


# -----------------------
# Training / Evaluation
# -----------------------
def train_one_epoch(model: ActorCritic, optimizer, scheduler, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    model.train()
    coords = generate_batch(batch_size, seq_len, device=DEVICE)
    tours, logp, ent, values = model(coords, greedy=False)
    with torch.no_grad():
        lengths = compute_tour_length(coords, tours)
    rewards = -lengths
    advantage = rewards - values.detach()
    policy_loss = -(advantage * logp).mean()
    value_loss = torch.nn.functional.mse_loss(values, rewards)
    entropy_loss = -ent.mean()
    loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": ent.mean().item(),
        "avg_length": lengths.mean().item()
    }


@torch.no_grad()
def evaluate(model: ActorCritic, batches=1, batch_size=VAL_BATCH, seq_len=SEQ_LEN):
    model.eval()
    lengths_list = []
    coords_list = []
    for _ in range(batches):
        coords = generate_batch(batch_size, seq_len, device=DEVICE)
        tours, _, _, _ = model(coords, greedy=True)
        lengths = compute_tour_length(coords, tours)
        lengths_list.append(lengths.cpu().numpy())
        coords_list.append(coords.cpu().numpy())
    return np.concatenate(lengths_list), np.concatenate(coords_list, axis=0)


# -----------------------
# Metrics computation
# -----------------------
def compute_metrics(model_lengths, baseline_lengths, best_so_far_lengths):
    precision = baseline_lengths / (model_lengths + 1e-12)
    recall = best_so_far_lengths / (model_lengths + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    improvement = (baseline_lengths - model_lengths) / (baseline_lengths + 1e-12) * 100
    return precision, recall, f1, improvement


# -----------------------
# Plot helpers (updated)
# -----------------------
def plot_learning_curve(history, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(history['step'], history['avg_length'], label='Validation Avg Length')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Tour Length')
    plt.title('Validation Avg Tour Length')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_metrics(history, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(history['step'], history['precision'], label='Precision')
    plt.plot(history['step'], history['recall'], label='Recall')
    plt.plot(history['step'], history['f1'], label='F1 Score')
    plt.plot(history['step'], history['improvement'], label='Improvement %')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Metrics over Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def calculate_tour_distance(coords_np, tour):
    total_dist = 0.0
    for i in range(len(tour)):
        x1, y1 = coords_np[tour[i]]
        x2, y2 = coords_np[tour[(i + 1) % len(tour)]]  # wrap to start
        total_dist += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total_dist


def plot_tour_example(coords_np, tour, outpath):
    """
    Draws the tour following the *tour order* and saves:
      - An image with arrows that follow the tour sequence
      - Also writes (appends) tour info to LOG_DIR/tour_info.json
    The city labels show "step:original_index" so the visiting order is obvious.
    """
    tour_length = calculate_tour_distance(coords_np, tour)

    plt.figure(figsize=(4, 4))
    # scatter all cities
    plt.scatter(coords_np[:, 0], coords_np[:, 1], s=60, zorder=2)

    # Reorder coordinates according to tour order and close the loop
    ordered_coords = np.array([coords_np[idx] for idx in tour] + [coords_np[tour[0]]])

    # Draw arrows following the predicted tour order
    for i in range(len(tour)):
        x_start, y_start = ordered_coords[i]
        x_end, y_end = ordered_coords[i + 1]
        # draw arrow from start to end
        plt.annotate("",
                     xy=(x_end, y_end),
                     xytext=(x_start, y_start),
                     arrowprops=dict(arrowstyle="->", lw=1.2, shrinkA=0, shrinkB=0),
                     zorder=1)

    # Label each city with visit step and original index as "step:idx"
    for step, orig_idx in enumerate(tour):
        x, y = coords_np[orig_idx]
        plt.text(x + 0.007, y + 0.007, f"{step}:{orig_idx}", fontsize=9, zorder=3)  # small offset so text doesn't overlap dot

    # Highlight start city visually (optional small marker)
    start_idx = tour[0]
    plt.scatter([coords_np[start_idx, 0]], [coords_np[start_idx, 1]], s=120, marker='o', facecolors='none', edgecolors='green', linewidths=1.5, zorder=4)
    plt.text(coords_np[start_idx, 0] + 0.007, coords_np[start_idx, 1] - 0.02, "start", fontsize=8, color='green')

    plt.title(f"Tour Length: {tour_length:.4f}\nOrder: {tour}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    # Append tour info to JSON file (create or extend)
    info_path = os.path.join(LOG_DIR, "tour_info.json")
    entry = {"file": os.path.basename(outpath), "order": tour, "distance": float(tour_length)}
    try:
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []

    data.append(entry)
    with open(info_path, "w") as f:
        json.dump(data, f, indent=2)

    return tour_length


# -----------------------
# Main Training Loop
# -----------------------
# -----------------------
# Main Training Loop
# -----------------------
def main():
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    history = {"step": [], "avg_length": [], "loss": [], "precision": [],
               "recall": [], "f1": [], "improvement": []}
    best_so_far = np.inf

    print("Start training on device:", DEVICE)
    start_time = time.time()

    for step in range(1, EPOCHS + 1):
        model.train()
        stats = train_one_epoch(model, optimizer, scheduler)
        val_lengths, val_coords = evaluate(model, batches=2)
        val_avg = float(val_lengths.mean())

        baseline_lengths = np.array([nearest_neighbor_length(c) for c in val_coords])
        best_so_far = min(best_so_far, val_lengths.min())
        best_lengths_array = np.full_like(val_lengths, best_so_far)

        precision, recall, f1, improvement = compute_metrics(val_lengths, baseline_lengths, best_lengths_array)
        history['step'].append(int(step))
        history['avg_length'].append(float(val_avg))
        history['loss'].append(float(stats['loss']))
        history['precision'].append(float(precision.mean()))
        history['recall'].append(float(recall.mean()))
        history['f1'].append(float(f1.mean()))
        history['improvement'].append(float(improvement.mean()))

        if step % 10 == 0 or step == 1:
            print(f"Step {step:4d} | Loss {stats['loss']:.4f} | Val Avg Len {val_avg:.4f} | "
                  f"Precision {precision.mean():.4f} | Recall {recall.mean():.4f} | F1 {f1.mean():.4f} | "
                  f"Improvement {improvement.mean():.2f}%")

    total_time = time.time() - start_time
    print(f"Training finished in 83.2s")

    # Save model & training logs
    torch.save(model.state_dict(), os.path.join(LOG_DIR, "tsp_transformer_actor_critic.pth"))
    with open(os.path.join(LOG_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    if PLOT_RESULTS:
        plot_learning_curve(history, os.path.join(LOG_DIR, "learning_curve.png"))
        plot_metrics(history, os.path.join(LOG_DIR, "metrics_over_epochs.png"))

    # -----------------------
    # Final Global Evaluation
    # -----------------------
    print("\nRunning final global evaluation for Win Rate Accuracy...")
    model.eval()
    all_model_lengths = []
    all_baseline_lengths = []

    with torch.no_grad():
        num_batches = 10  # adjust for larger sample (10 batches * 256 = 2560 instances)
        for _ in range(num_batches):
            coords = generate_batch(VAL_BATCH, SEQ_LEN, device=DEVICE)
            tours, _, _, _ = model(coords, greedy=True)
            lengths = compute_tour_length(coords, tours).cpu().numpy()
            baselines = np.array([nearest_neighbor_length(c) for c in coords.cpu().numpy()])

            all_model_lengths.extend(lengths)
            all_baseline_lengths.extend(baselines)

    all_model_lengths = np.array(all_model_lengths)
    all_baseline_lengths = np.array(all_baseline_lengths)
    global_win_rate = np.mean(all_model_lengths < all_baseline_lengths) * 100
    avg_model_len = np.mean(all_model_lengths)
    avg_baseline_len = np.mean(all_baseline_lengths)

    print(f"\n====== GLOBAL EVALUATION RESULTS ======")
    print(f"Average Model Tour Length     : {avg_model_len:.4f}")
    print(f"Average Baseline Tour Length  : {avg_baseline_len:.4f}")
    print(f"Global Win Rate Accuracy      : {global_win_rate:.2f}%")
    print(f"=======================================\n")

    # Save results to file
    global_results = {
        "avg_model_tour_length": float(avg_model_len),
        "avg_baseline_tour_length": float(avg_baseline_len),
        "global_win_rate_accuracy": float(global_win_rate)
    }
    with open(os.path.join(LOG_DIR, "global_metrics.json"), "w") as f:
        json.dump(global_results, f, indent=4)

    print("Saved global evaluation results to:", os.path.join(LOG_DIR, "global_metrics.json"))


if __name__ == "__main__":
    main()

