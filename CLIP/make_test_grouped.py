import pickle
from collections import defaultdict

# 경로 설정
test_data_path = "/home/lej/MMKG-PF/data/processed_random/test_data.pkl"
output_path = "/home/lej/MMKG-PF/data/processed_random/test_grouped.pkl"

# 불러오기
with open(test_data_path, "rb") as f:
    test_data = pickle.load(f)

# 유저 기준 group
grouped = defaultdict(list)
for u, i in test_data:
    grouped[u].append(i)

# 저장
with open(output_path, "wb") as f:
    pickle.dump(dict(grouped), f)

print(f"test_grouped.pkl 생성 완료: {output_path} (유저 수: {len(grouped)})")