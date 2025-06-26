import torch, pickle, random, numpy as np, os
from torch.utils.data import Dataset, DataLoader
from bpr_clip import BPR_MF_CLIP_Projected
import torch.optim as optim
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils  import setup_logger, bpr_loss, BPRDataset, plot_metrics
from collections import defaultdict


def evaluate(model, val_data, train_data, num_items, K=10):
    device = model.user_emb.weight.device

    user_pos = defaultdict(set)
    for u, i in train_data:
        user_pos[u].add(i)

    val_grouped = defaultdict(list)
    for u, i in val_data:
        val_grouped[u].append(i)

    hit_list, ndcg_list, recall_list = [], [], []
    model.eval()
    with torch.no_grad():
        for u, gt_items in val_grouped.items():
            gt_items = set(gt_items)
            if len(gt_items) == 0:
                continue

            candidates = list(set(range(num_items)) - user_pos[u])
            if len(candidates) == 0:
                continue

            u_tensor = torch.LongTensor([u] * len(candidates)).to(device)
            i_tensor = torch.LongTensor(candidates).to(device)

            u_emb = model.user_emb(u_tensor)
            i_emb = model.proj_img(model.clip_img_emb(i_tensor))
            if model.use_text:
                i_emb += model.proj_txt(model.clip_txt_emb(i_tensor))

            scores = torch.sum(u_emb * i_emb, dim=1)
            ranked_items = [candidates[i] for i in torch.argsort(scores, descending=True)[:K]]

            hits = len(set(ranked_items) & gt_items)
            hit_list.append(1 if hits > 0 else 0)
            recall_list.append(hits / len(gt_items))

            dcg = sum([1 / np.log2(ranked_items.index(i) + 2) for i in gt_items if i in ranked_items])
            ideal_len = min(len(gt_items), K)
            idcg = sum([1 / np.log2(i + 2) for i in range(ideal_len)])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)

    return np.mean(hit_list), np.mean(ndcg_list), np.mean(recall_list)



def train(epochs, use_text=True):
    setup_logger(f"results/log_bpr_clip_{epochs}.txt")

    with open("/home/lej/MMKG-PF/data/processed_random/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("/home/lej/MMKG-PF/data/processed_random/val_data.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open("/home/lej/MMKG-PF/data/processed_random/mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    item2idx = mapping["item2idx"]
    num_users = len(mapping["user2idx"])
    num_items = len(item2idx)

    dataset = BPRDataset(train_data, num_items)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = BPR_MF_CLIP_Projected(
        num_users=num_users,
        item2idx=item2idx,
        clip_image_path="/home/lej/MMKG-PF/data/clip_image_embeds.npy",
        clip_text_path="/home/lej/MMKG-PF/data/clip_text_embeds.npy",
        proj_dim=128,
        use_text=use_text,
        freeze=False
    ).to(device)

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
            hit, ndcg, recall = evaluate(model, val_data, train_data, num_items, K=10)
            hit_list.append(hit)
            ndcg_list.append(ndcg)
            recall_list.append(recall)
            print(f"Validation @ Epoch {epoch} → Hit@10: {hit:.4f}, nDCG@10: {ndcg:.4f}, Recall@10: {recall:.4f}")

    # 훈련 완료 후 마지막 모델 저장
    torch.save(model.state_dict(), "results/best_model.pt")
    print("Final model saved after training.")
    plot_metrics(loss_list, hit_list, ndcg_list, recall_list)


if __name__ == "__main__":
    train(1000, use_text=True)