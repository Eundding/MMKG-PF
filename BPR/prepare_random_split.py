import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split

def load_and_prepare_random_split():
    ratings = pd.read_csv("/home/lej/MMKG-PF/data/ml-1m/ratings.dat", sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"], encoding="latin1")
    meta = pd.read_csv("/home/lej/MMKG-PF/data/movie_with_plot_poster.csv")

    # 필터링: multimodal 정보 + 평점 3점 이상
    ratings = ratings[ratings['movieId'].isin(meta['movieId'])]
    ratings = ratings[ratings['rating'] >= 3.0]

    # 사용자 필터링: 최소 5개 이상 interaction
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    ratings = ratings[ratings['userId'].isin(valid_users)]

    # 인덱스 매핑
    user_ids = ratings['userId'].unique()
    item_ids = ratings['movieId'].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}
    ratings['user_idx'] = ratings['userId'].map(user2idx)
    ratings['item_idx'] = ratings['movieId'].map(item2idx)

    # 80% train_full, 20% test
    train_full, test = train_test_split(ratings, test_size=0.2, random_state=42)
    # train_full 안에서 다시 10%를 validation으로
    train, val = train_test_split(train_full, test_size=0.1, random_state=42)

    # 데이터 저장
    out_path = "/home/lej/MMKG-PF/data/processed_random"
    os.makedirs(out_path, exist_ok=True)
    for name, data in zip(['train', 'val', 'test'], [train, val, test]):
        data_pairs = data[['user_idx', 'item_idx']].drop_duplicates().values.tolist()
        with open(f"{out_path}/{name}_data.pkl", "wb") as f:
            pickle.dump(data_pairs, f)

    with open(f"{out_path}/mapping.pkl", "wb") as f:
        pickle.dump({'user2idx': user2idx, 'item2idx': item2idx}, f)

    print("Split 완료: train/val/test 데이터 저장됨.")

if __name__ == "__main__":
    load_and_prepare_random_split()