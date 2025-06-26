import torch
import pickle
import numpy as np
import random
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils  import setup_logger, hit_at_k, recall_at_k, ndcg_at_k
from bpr_clip import BPR_MF_CLIP_Projected


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def evaluate(model_path="results/best_model.pt", eval_set="test", K_list=[5, 10, 20]):
    setup_logger(f"results/eval_log_bpr_clip_{eval_set}.txt")

    base_path = "/home/lej/MMKG-PF/data/processed_random"
    with open(f"{base_path}/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"{base_path}/{eval_set}_grouped.pkl", "rb") as f:
        eval_data = pickle.load(f)
    with open(f"{base_path}/mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    item2idx = mapping['item2idx']
    num_users = len(mapping['user2idx'])
    num_items = len(item2idx)

    user_pos = {u: set() for u in range(num_users)}
    for u, i in train_data:
        user_pos[u].add(i)

    model = BPR_MF_CLIP_Projected(
        num_users=num_users,
        item2idx=item2idx,
        clip_image_path="/home/lej/MMKG-PF/data/clip_image_embeds.npy",
clip_text_path="/home/lej/MMKG-PF/data/clip_text_embeds.npy", 
        proj_dim=128,
        use_text=True,  # 텍스트도 쓰면 True
        freeze=False
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    results = {k: {'hit': [], 'ndcg': [], 'recall': []} for k in K_list}

    for u, true_items in tqdm(eval_data.items(), desc=f"Evaluating ({eval_set})"):
        candidates = list(set(range(num_items)) - user_pos[u])
        if len(candidates) == 0 or len(true_items) == 0:
            continue

        u_tensor = torch.LongTensor([u] * len(candidates))
        i_tensor = torch.LongTensor(candidates)

        with torch.no_grad():
            u_vec = model.user_emb(u_tensor)
            i_emb = model.proj_img(model.clip_img_emb(i_tensor))
            if model.use_text:
                i_emb += model.proj_txt(model.clip_txt_emb(i_tensor))
            scores = torch.sum(u_vec * i_emb, dim=1)

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