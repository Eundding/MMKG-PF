import torch
import torch.nn as nn
import numpy as np

class BPR_MF_CLIP_Projected(nn.Module):
    def __init__(self, num_users, item2idx, clip_image_path, clip_text_path=None,
                 clip_dim=512, proj_dim=64, freeze=False, use_text=False):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, proj_dim)
        self.use_text = use_text

        def load_clip_embedding(path):
            clip_dict = np.load(path, allow_pickle=True).item()
            matrix = np.zeros((len(item2idx), clip_dim), dtype=np.float32)
            for movie_id, idx in item2idx.items():
                matrix[idx] = clip_dict.get(movie_id, np.random.normal(scale=0.01, size=(clip_dim,)))
            return nn.Embedding.from_pretrained(torch.tensor(matrix), freeze=freeze)

        self.clip_img_emb = load_clip_embedding(clip_image_path)
        if self.use_text:
            self.clip_txt_emb = load_clip_embedding(clip_text_path)

        self.proj_img = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        if self.use_text:
            self.proj_txt = nn.Sequential(
                nn.Linear(clip_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, u, i, j):
        u_e = self.user_emb(u)

        i_img = self.proj_img(self.clip_img_emb(i))
        j_img = self.proj_img(self.clip_img_emb(j))

        if self.use_text:
            i_txt = self.proj_txt(self.clip_txt_emb(i))
            j_txt = self.proj_txt(self.clip_txt_emb(j))
            i_e = i_img + i_txt
            j_e = j_img + j_txt
        else:
            i_e = i_img
            j_e = j_img

        return torch.sum(u_e * (i_e - j_e), dim=1)