import os, torch, numpy as np, pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import requests
import torch.nn.functional as F 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df = pd.read_csv("/home/lej/MMKG-PF/data/movie_with_plot_poster.csv")
img_embeds, txt_embeds = {}, {}

os.makedirs("clip_posters", exist_ok=True)

for idx, row in df.iterrows():
    movie_id = row["movieId"]
    plot = str(row["plot"])
    image_url = row["poster"]

    try:
        # 이미지 처리
        img_path = f"clip_posters/{movie_id}.jpg"
        if not os.path.exists(img_path):
            img = Image.open(BytesIO(requests.get(image_url, timeout=10).content)).convert("RGB")
            img.save(img_path)
        else:
            img = Image.open(img_path).convert("RGB")

        # 이미지 임베딩 추출 + 정규화
        img_input = processor(images=img, return_tensors="pt").to(device)
        img_feat = F.normalize(model.get_image_features(**img_input), dim=-1) 
        img_embeds[movie_id] = img_feat.squeeze().detach().cpu().numpy()                

        # 텍스트 임베딩 추출 + 정규화
        txt_input = processor(text=[plot], return_tensors="pt", padding=True).to(device)
        txt_feat = F.normalize(model.get_text_features(**txt_input), dim=-1)  
        txt_embeds[movie_id] = txt_feat.squeeze().detach().cpu().numpy()             

    except Exception as e:
        print(f"[{movie_id}] 실패: {e}")

np.save("/home/lej/MMKG-PF/data/clip_image_embeds.npy", img_embeds)
np.save("/home/lej/MMKG-PF/data/clip_text_embeds.npy", txt_embeds)
print("이미지 + 텍스트 CLIP 임베딩 저장 완료 (정규화 포함)")