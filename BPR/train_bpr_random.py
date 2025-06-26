import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import numpy as np
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils  import setup_logger, bpr_loss, BPRDataset, plot_metrics
from collections import defaultdict
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class BPR_MF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01) 
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i, j):
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        j_e = self.item_emb(j)
        return torch.sum(u_e * (i_e - j_e), dim=1)

def evaluate_on_val(model, val_data, train_data, num_items, K=10):
    device = model.user_emb.weight.device

    # 사용자별 정답 묶기
    val_grouped = defaultdict(set)
    for u, i in val_data:
        val_grouped[u].add(i)

    user_pos = defaultdict(set)
    for u, i in train_data:
        user_pos[u].add(i)

    hit_list, ndcg_list, recall_list = [], [], []
    model.eval()
    with torch.no_grad():
        for u, gt_items in val_grouped.items():
            if len(gt_items) == 0:
                continue

            candidates = list(set(range(num_items)) - user_pos[u])
            if len(candidates) == 0:
                continue

            u_tensor = torch.LongTensor([u] * len(candidates)).to(device)
            i_tensor = torch.LongTensor(candidates).to(device)

            scores = torch.sum(model.user_emb(u_tensor) * model.item_emb(i_tensor), dim=1)
            ranked_items = [candidates[i] for i in torch.argsort(scores, descending=True)[:K]]

            # Hit@K
            hits = len(set(ranked_items) & gt_items)
            hit_list.append(1 if hits > 0 else 0)

            # Recall@K
            recall = hits / len(gt_items)
            recall_list.append(recall)

            # nDCG@K
            dcg = sum([1 / np.log2(ranked_items.index(i) + 2) for i in gt_items if i in ranked_items])
            ideal_len = min(len(gt_items), K)
            idcg = sum([1 / np.log2(i + 2) for i in range(ideal_len)])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)

    return np.mean(hit_list), np.mean(ndcg_list), np.mean(recall_list)

def train(epochs):
    os.makedirs("results", exist_ok=True)
    setup_logger(f"results/log_bpr_{epochs}.txt")

    # Load data
    with open("/home/lej/MMKG-PF/data/processed_random/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("/home/lej/MMKG-PF/data/processed_random/val_data.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open("/home/lej/MMKG-PF/data/processed_random/mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    num_users = len(mapping["user2idx"])
    num_items = len(mapping["item2idx"])

    dataset = BPRDataset(train_data, num_items)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = BPR_MF(num_users, num_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_list = []
    hit_list = []
    ndcg_list = []
    recall_list = []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for u, i, j in loader:
            u, i, j = u.squeeze().to(device), i.squeeze().to(device), j.squeeze().to(device)
            score = model(u, i, j)
            loss = bpr_loss(score, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")
        loss_list.append(total_loss)

        if epoch % 5 == 0:
            hit, ndcg, recall = evaluate_on_val(model, val_data, train_data, num_items)
            hit_list.append(hit)
            ndcg_list.append(ndcg)
            recall_list.append(recall)
            print(f"Validation Hit@10: {hit:.4f}, nDCG@10: {ndcg:.4f}, Recall@10: {recall:.4f}")

    torch.save(model.state_dict(), "results/best_model.pt")
    print("Final model saved after training.")
    plot_metrics(loss_list, hit_list, ndcg_list, recall_list)

if __name__ == "__main__":
    train(1000)