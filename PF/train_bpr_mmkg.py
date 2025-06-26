import torch, pickle, os, random, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils  import setup_logger, bpr_loss, BPRDataset, plot_metrics
from tqdm import tqdm
from collections import defaultdict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class BPR_MF_MMKG_PF(nn.Module):
    def __init__(self, num_users, item2idx, img_path, txt_path, kg_path, emb_dim=512):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_items = len(item2idx)

        # User embeddings & personalized fusion weights
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.alpha = nn.Embedding(num_users, 3)  # a_img, a_txt, a_kg
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.alpha.weight, std=0.01)

        # Load modality embeddings
        clip_img = np.load(img_path, allow_pickle=True).item()
        clip_txt = np.load(txt_path, allow_pickle=True).item()
        kg_embeds = np.load(kg_path, allow_pickle=True).item()

        # Initialize fixed item embeddings
        self.item_embed_img = torch.zeros(self.num_items, emb_dim)
        self.item_embed_txt = torch.zeros(self.num_items, emb_dim)
        self.item_embed_kg = torch.zeros(self.num_items, emb_dim)

        for m_id, idx in item2idx.items():
            if m_id in clip_img:
                self.item_embed_img[idx] = torch.tensor(clip_img[m_id], dtype=torch.float32)
            if m_id in clip_txt:
                self.item_embed_txt[idx] = torch.tensor(clip_txt[m_id], dtype=torch.float32)
            if m_id in kg_embeds:
                self.item_embed_kg[idx] = torch.tensor(kg_embeds[m_id], dtype=torch.float32)

        # Freeze pre-trained item embeddings
        self.item_embed_img = nn.Parameter(self.item_embed_img, requires_grad=False)
        self.item_embed_txt = nn.Parameter(self.item_embed_txt, requires_grad=False)
        self.item_embed_kg = nn.Parameter(self.item_embed_kg, requires_grad=False)

    def forward(self, u, i, j):
        u_e = self.user_emb(u)                     # (batch_size, emb_dim)
        alpha = torch.softmax(self.alpha(u), dim=1)  # (batch_size, 3)

        # Positive
        i_img = self.item_embed_img[i]  # (batch_size, emb_dim)
        i_txt = self.item_embed_txt[i]
        i_kg  = self.item_embed_kg[i]
        # Negative
        j_img = self.item_embed_img[j]
        j_txt = self.item_embed_txt[j]
        j_kg  = self.item_embed_kg[j]

        # Personalized item fusion
        i_e = (alpha[:, 0:1] * i_img +
               alpha[:, 1:2] * i_txt +
               alpha[:, 2:3] * i_kg)

        j_e = (alpha[:, 0:1] * j_img +
               alpha[:, 1:2] * j_txt +
               alpha[:, 2:3] * j_kg)

        # BPR score
        return torch.sum(u_e * (i_e - j_e), dim=1)


# validation 평가
def evaluate_on_val(model, val_data, train_data, num_items, K=10):
    device = model.user_emb.weight.device

    # 사용자별 정답 묶기
    val_grouped = defaultdict(set)
    for u, i in val_data:
        val_grouped[u].add(i)

    # 사용자별 학습 아이템
    user_pos = defaultdict(set)
    for u, i in train_data:
        user_pos[u].add(i)

    hit_list, ndcg_list, recall_list = [], [], []
    model.eval()
    with torch.no_grad():
        for u, gt_items in val_grouped.items():
            if len(gt_items) == 0:
                continue

            # 전체 아이템 중에서 유저가 훈련에서 본 적 없는 아이템만 후보로 
            candidates = list(set(range(num_items)) - user_pos[u])
            if len(candidates) == 0:
                continue

            u_tensor = torch.LongTensor([u] * len(candidates)).to(device)
            i_tensor = torch.LongTensor(candidates).to(device)

            # 사용자별 α 가중치 계산
            alpha = torch.softmax(model.alpha(torch.tensor([u]).to(device)), dim=1)  # (1, 3)
            img_vecs = model.item_embed_img[i_tensor]  # (N, D)
            txt_vecs = model.item_embed_txt[i_tensor]
            kg_vecs = model.item_embed_kg[i_tensor]

            # 개인화 융합된 item 임베딩 계산
            item_vecs = (
                alpha[0, 0] * img_vecs +
                alpha[0, 1] * txt_vecs +
                alpha[0, 2] * kg_vecs
            )

            # 유저 벡터
            # 모든 후보 아이템과 사용자 벡터 간 내적 → 추천 점수 계산
            user_vec = model.user_emb(torch.tensor([u]).to(device))  # (1, D)
            scores = torch.matmul(item_vecs, user_vec.squeeze(0))  # (N,)

            # 상위 K개 예측
            topk_indices = torch.topk(scores, K).indices
            ranked_items = [candidates[i] for i in topk_indices]

            # Hit@K 정답 아이템이 K개 중에 하나라도 있으면 1, 아니면 0
            hits = len(set(ranked_items) & gt_items)
            hit_list.append(1 if hits > 0 else 0)

            # Recall@K 정답 중 몇 개가 K개 추천 안에 들어갔는지 비율
            recall = hits / len(gt_items)
            recall_list.append(recall)

            """nDCG@K
            DCG: 정답이 몇 번째에 등장했는지에 따라 가중치 부여
            IDCG: 이상적인 DCG (정답들이 앞에 나올 때)
            nDCG = DCG / IDCG"""
            dcg = sum([1 / np.log2(ranked_items.index(i) + 2) for i in gt_items if i in ranked_items])
            ideal_len = min(len(gt_items), K)
            idcg = sum([1 / np.log2(i + 2) for i in range(ideal_len)])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)

    return np.mean(hit_list), np.mean(ndcg_list), np.mean(recall_list)

def train(epochs):
    os.makedirs("results", exist_ok=True)
    setup_logger(f"results/log_bpr_mmkg_pf_{epochs}.txt")

    # Load data
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
    model = BPR_MF_MMKG_PF(
        num_users=num_users,
        item2idx=item2idx,
        img_path="/home/lej/MMKG-PF/data/clip_image_embeds.npy",
        txt_path="/home/lej/MMKG-PF/data/clip_text_embeds.npy",
        kg_path="/home/lej/MMKG-PF/PF/kg_embeds.npy" # kg_embeds 파일 
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Metrics log
    loss_list, hit_list, ndcg_list, recall_list = [], [], [], []

    for epoch in range(1, epochs + 1):
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

        # Validation
        if epoch % 5 == 0:
            hit, ndcg, recall = evaluate_on_val(model, val_data, train_data, num_items)
            hit_list.append(hit)
            ndcg_list.append(ndcg)
            recall_list.append(recall)
            print(f"[Validation @ Epoch {epoch}] Hit@10: {hit:.4f}, nDCG@10: {ndcg:.4f}, Recall@10: {recall:.4f}")

    # Final save
    torch.save(model.state_dict(), f"results/best_model_mmkg_pf.pt")
    print("Final model saved.")

    # Plot training logs
    plot_metrics(loss_list, hit_list, ndcg_list, recall_list)


if __name__ == "__main__":
    train(1000)
