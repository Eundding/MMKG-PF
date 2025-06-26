import pandas as pd

# 파일 경로 설정
movies_dat_path = "/home/lej/MMKG-PF/data/ml-1m/movies.dat"
movie_meta_path = "/home/lej/MMKG-PF/data/movie_with_plot_poster.csv"
output_path = "/home/lej/MMKG-PF/data/triples_mmkg.csv"

# 장르 파일 불러오기
movies = pd.read_csv(movies_dat_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin1")

# 줄거리/포스터 파일 불러오기
meta = pd.read_csv(movie_meta_path)

# 공통된 movieId만 사용
common_ids = set(movies.movieId).intersection(set(meta.movieId))
movies = movies[movies.movieId.isin(common_ids)]
meta = meta[meta.movieId.isin(common_ids)]

triples = []

# 1. hasGenre triple 생성
for _, row in movies.iterrows():
    movie_id = row["movieId"]
    genres = row["genres"].split("|")
    for genre in genres:
        triples.append([movie_id, "hasGenre", genre])

# 2. hasPlot, hasPoster triple 생성
for _, row in meta.iterrows():
    movie_id = row["movieId"]
    triples.append([movie_id, "hasPlot", f"plot_{movie_id}"])
    triples.append([movie_id, "hasPoster", f"poster_{movie_id}"])

# 저장
triple_df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
triple_df.to_csv(output_path, index=False)

print(f"Triple 파일 저장 완료: {output_path} ({len(triple_df)} triples)")