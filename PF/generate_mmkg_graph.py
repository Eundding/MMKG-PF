import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import pickle

# 경로 설정
triple_path = "/home/lej/MMKG-PF/data/triples_mmkg.csv"
mapping_path = "/home/lej/MMKG-PF/data/mmkg_entity_mapping.pkl"
img_embed_path = "/home/lej/MMKG-PF/data/clip_image_embeds.npy"
txt_embed_path = "/home/lej/MMKG-PF/data/clip_text_embeds.npy"

output_graph_path = "/home/lej/MMKG-PF/data/mmkg_graph_multimodal.pt"

# 데이터 불러오기
triples = pd.read_csv(triple_path)
with open(mapping_path, "rb") as f:
    mapping = pickle.load(f)

entity2id = mapping["entity2id"]
n_nodes = len(entity2id)

# CLIP 임베딩 로드
img_embeds = np.load(img_embed_path, allow_pickle=True).item()
txt_embeds = np.load(txt_embed_path, allow_pickle=True).item()

# 노드 feature matrix 초기화
x = torch.zeros(n_nodes, 512)

# 각 노드에 CLIP 임베딩만 할당
for ent, idx in tqdm(entity2id.items(), desc="노드 임베딩 구성"):
    if ent.startswith("plot_"):
        movie_id = int(ent.split("_")[1])
        if movie_id in txt_embeds:
            x[idx] = torch.tensor(txt_embeds[movie_id], dtype=torch.float32)

    elif ent.startswith("poster_"):
        movie_id = int(ent.split("_")[1])
        if movie_id in img_embeds:
            x[idx] = torch.tensor(img_embeds[movie_id], dtype=torch.float32)

    else: # 장르, movieId 등은 랜덤 초기화
        x[idx] = torch.randn(512) * 0.01

# edge_index 생성
edge_list = []
for _, row in triples.iterrows():
    h = entity2id[str(row["head"])]
    t = entity2id[str(row["tail"])]
    edge_list.append([h, t])
    edge_list.append([t, h])  # 양방향

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() # [2, num_edges]로 변환

# 저장
torch.save({"x": x, "edge_index": edge_index}, output_graph_path)
print(f"멀티모달 MMKG 그래프 저장 완료: {output_graph_path}")
print(f"→ 노드 수: {n_nodes}, 엣지 수: {edge_index.size(1)}")