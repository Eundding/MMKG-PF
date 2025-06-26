#  GAT 구조
# 비지도 학습으로 영화 줄거리, 포스터 의미전파만(의미 강화)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pickle
import numpy as np

# 경로 설정
graph_path = "/home/lej/MMKG-PF/data/mmkg_graph_multimodal.pt"
mapping_path = "/home/lej/MMKG-PF/data/mmkg_entity_mapping.pkl"

# GAT 모델 정의
class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=True, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

# 데이터 로딩
graph = torch.load(graph_path)
with open(mapping_path, "rb") as f:
    mapping = pickle.load(f)

x = graph["x"]
edge_index = graph["edge_index"]
entity2id = mapping["entity2id"]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
x = x.to(device)
edge_index = edge_index.to(device)

# GAT 모델 초기화
model = GATEncoder(input_dim=512, hidden_dim=128, output_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 200
model.train()
for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = torch.norm(out) * 0 
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} completed.")

# 평가 모드 전환 후 임베딩 추출
model.eval()
with torch.no_grad():
    final_embeds = model(x, edge_index).cpu().numpy()

# 영화 노드만 필터링하여 저장
movie_kg_embeds = {}
for ent, idx in entity2id.items():
    if ent.isdigit():  # movieId는 숫자로만 구성되어 있음
        movie_kg_embeds[int(ent)] = final_embeds[idx]

# 저장
output_embed_path = f"/home/lej/MMKG-PF/PF/kg_embeds.npy"
np.save(output_embed_path, movie_kg_embeds)
print(f"MMKG 임베딩 저장 완료: {output_embed_path} (총 영화 수: {len(movie_kg_embeds)})")