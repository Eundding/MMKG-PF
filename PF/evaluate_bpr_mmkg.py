import torch
import numpy as np
import pickle
import random
import sys
from tqdm import tqdm
from train_bpr_mmkg import BPR_MF_MMKG_PF  # 같은 모델 구조 사용

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def setup_logger(log_path):
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger()

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

def evaluate(model_path="results/best_model_mmkg_pf.pt", eval_set="test", K_list=[5, 10, 20]):
    setup_logger(f"results/eval_log_bpr_mmkg_pf_{eval_set}.txt")

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

    model = BPR_MF_MMKG_PF(
        num_users=num_users,
        item2idx=item2idx,
        img_path="/home/lej/MMKG-PF/data/clip_image_embeds.npy",
        txt_path="/home/lej/MMKG-PF/data/clip_text_embeds.npy",
        kg_path="/home/lej/MMKG-PF/PF/kg_embeds_500.npy"
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    results = {k: {'hit': [], 'ndcg': [], 'recall': []} for k in K_list}

    for u, true_items in tqdm(eval_data.items(), desc=f"Evaluating ({eval_set})"):
        neg_items = list(set(range(num_items)) - user_pos[u] - set(true_items))
        if len(neg_items) < 100:
            continue
        sampled_neg = np.random.choice(neg_items, 100, replace=False).tolist()
        candidates = sampled_neg + true_items

        u_tensor = torch.LongTensor([u] * len(candidates))
        i_tensor = torch.LongTensor(candidates)

        with torch.no_grad():
            alpha = torch.softmax(model.alpha(torch.tensor([u])), dim=1)
            item_embed = (alpha[0, 0] * model.item_embed_img[i_tensor] +
                          alpha[0, 1] * model.item_embed_txt[i_tensor] +
                          alpha[0, 2] * model.item_embed_kg[i_tensor])
            user_vec = model.user_emb(torch.tensor([u])).repeat(len(candidates), 1)
            scores = torch.sum(user_vec * item_embed, dim=1)

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
