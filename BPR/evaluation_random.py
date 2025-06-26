import torch
import pickle
import numpy as np
import random
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import setup_logger
from train_bpr_random import BPR_MF  # baseline 모델 사용

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def hit_at_k(ranked_list, ground_truth, k):
    return int(any(item in ranked_list[:k] for item in ground_truth))

def ndcg_at_k(ranked_list, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_list, ground_truth, k):
    hit = sum(1 for item in ground_truth if item in ranked_list[:k])
    return hit / len(ground_truth) if len(ground_truth) > 0 else 0.0

def evaluate(model_path="results/best_model.pt", eval_set="test", K_list=[5, 10, 20]):
    setup_logger(f"results/eval_log_bpr_random_{eval_set}.txt")

    base_path = "/home/lej/MMKG-PF/data/processed_random"
    with open(f"{base_path}/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"{base_path}/{eval_set}_grouped.pkl", "rb") as f:
        eval_data = pickle.load(f)
    with open(f"{base_path}/mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    num_users = len(mapping['user2idx'])
    num_items = len(mapping['item2idx'])

    user_pos = {u: set() for u in range(num_users)}
    for u, i in train_data:
        user_pos[u].add(i)

    model = BPR_MF(num_users, num_items)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    results = {k: {'hit': [], 'ndcg': [], 'recall': []} for k in K_list}

    for u, true_items in tqdm(eval_data.items(), desc=f"Evaluating ({eval_set})"):
        # ⚠️ Full ranking: 후보 아이템은 유저가 본 적 없는 전체 아이템
        candidates = list(set(range(num_items)) - user_pos[u])
        if len(candidates) < 1 or len(true_items) == 0:
            continue

        u_tensor = torch.LongTensor([u] * len(candidates))
        i_tensor = torch.LongTensor(candidates)

        with torch.no_grad():
            user_vec = model.user_emb(u_tensor)
            item_vec = model.item_emb(i_tensor)
            scores = torch.sum(user_vec * item_vec, dim=1)

        ranked_idx = torch.argsort(scores, descending=True).tolist()
        ranked_items = [candidates[i] for i in ranked_idx]

        for k in K_list:
            results[k]['hit'].append(hit_at_k(ranked_items, true_items, k))
            results[k]['ndcg'].append(ndcg_at_k(ranked_items, true_items, k))
            results[k]['recall'].append(recall_at_k(ranked_items, true_items, k))

    for k in K_list:
        print(f"\n[EVAL on {eval_set.upper()} @ K={k}]")
        print(f"Hit@{k}:     {np.mean(results[k]['hit']):.4f}")
        print(f"nDCG@{k}:    {np.mean(results[k]['ndcg']):.4f}")
        print(f"Recall@{k}:  {np.mean(results[k]['recall']):.4f}")

if __name__ == "__main__":
    evaluate()
