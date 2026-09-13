
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Smart Movie Recommender", page_icon="🎬", layout="wide")

movies = pd.DataFrame([
    [1,"Interstellar","Sci-Fi Space Adventure","space astronaut nasa black hole future survival science"],
    [2,"The Martian","Sci-Fi Space Adventure","mars astronaut space survival nasa science engineering"],
    [3,"Gravity","Sci-Fi Space Thriller","space astronaut survival gravity mission nasa"],
    [4,"Inception","Sci-Fi Thriller","dream mind technology thriller heist action"],
    [5,"The Matrix","Sci-Fi Action","technology virtual reality artificial intelligence action"],
    [6,"Avengers","Action Superhero","superhero marvel action team adventure technology"],
    [7,"Iron Man","Action Superhero","superhero technology iron man marvel action"],
    [8,"Titanic","Romance Drama","romance love ship historical drama tragedy"],
    [9,"The Notebook","Romance Drama","romance love relationship emotional drama"],
    [10,"The Dark Knight","Action Crime","batman superhero crime joker action thriller"],
    [11,"Toy Story","Animation Comedy","animation friendship toys family comedy adventure"],
    [12,"Finding Nemo","Animation Adventure","animation ocean family friendship adventure"],
], columns=["movie_id","title","genre","description"])

ratings = pd.DataFrame([
    [1,1,5],[1,2,5],[1,3,4],[1,4,4],[1,5,5],
    [2,1,5],[2,2,4],[2,3,5],[2,4,3],[2,6,4],
    [3,4,5],[3,5,5],[3,6,4],[3,7,5],[3,10,5],
    [4,8,5],[4,9,5],[4,11,3],[4,12,3],
    [5,6,5],[5,7,5],[5,10,4],[5,4,3],[5,5,4],
    [6,11,5],[6,12,5],[6,8,3],[6,9,3],
    [7,1,4],[7,2,5],[7,4,5],[7,5,4],[7,10,4],
    [8,8,4],[8,9,5],[8,11,4],[8,12,5],
], columns=["user_id","movie_id","rating"])

# ---------------- CONTENT MODEL (CO2) ----------------
tfidf = TfidfVectorizer(stop_words="english")
content_matrix = tfidf.fit_transform(movies["genre"] + " " + movies["description"])
content_similarity = cosine_similarity(content_matrix)

# ---------------- COLLABORATIVE MODEL (CO1) ----------------
user_item = ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
item_user = user_item.T

# Item-based similarity
item_similarity = cosine_similarity(item_user)
item_similarity_df = pd.DataFrame(item_similarity, index=item_user.index, columns=item_user.index)

n_components = min(3, min(user_item.shape) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item)
reconstructed = np.dot(user_factors, svd.components_)
mf_scores = pd.DataFrame(reconstructed, index=user_item.index, columns=user_item.columns)

def normalize_scores(series):
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def collaborative_recommend(user_id, n=5):
    rated = ratings[ratings.user_id == user_id]
    rated_ids = set(rated.movie_id)

    item_scores = pd.Series(0.0, index=movies.movie_id, dtype=float)
    weights = pd.Series(0.0, index=movies.movie_id, dtype=float)

    for _, row in rated.iterrows():
        mid = int(row.movie_id)
        for target in movies.movie_id:
            if target == mid:
                continue
            sim = item_similarity_df.loc[mid, target]
            item_scores[target] += sim * row.rating
            weights[target] += abs(sim)

    item_scores = item_scores / weights.replace(0, np.nan)
    item_scores = item_scores.fillna(0)

    mf = mf_scores.loc[user_id].reindex(movies.movie_id).fillna(0)
    combined = 0.5 * normalize_scores(item_scores) + 0.5 * normalize_scores(mf)
    combined.loc[list(rated_ids)] = -1
    return combined.sort_values(ascending=False).head(n)

# ---------------- CONTENT RECOMMENDER (CO2) ----------------
def content_recommend(movie_id, n=5):
    idx = movies.index[movies.movie_id == movie_id][0]
    sims = pd.Series(content_similarity[idx], index=movies.movie_id)
    sims.loc[movie_id] = -1
    return sims.sort_values(ascending=False).head(n)

def user_content_scores(user_id):
    rated = ratings[ratings.user_id == user_id]
    scores = pd.Series(0.0, index=movies.movie_id)
    weights = pd.Series(0.0, index=movies.movie_id)

    for _, row in rated.iterrows():
        mid = int(row.movie_id)
        idx = movies.index[movies.movie_id == mid][0]
        for target in movies.movie_id:
            if target == mid:
                continue
            sim = content_similarity[idx, movies.index[movies.movie_id == target][0]]
            scores[target] += sim * row.rating
            weights[target] += abs(sim)

    scores = scores / weights.replace(0, np.nan)
    return scores.fillna(0)

@st.cache_resource
def train_ncf():
    try:
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense

        user_ids = sorted(ratings.user_id.unique())
        movie_ids = sorted(movies.movie_id.unique())
        u_map = {u:i for i,u in enumerate(user_ids)}
        m_map = {m:i for i,m in enumerate(movie_ids)}

        X_user = np.array([u_map[u] for u in ratings.user_id])
        X_movie = np.array([m_map[m] for m in ratings.movie_id])
        y = ratings.rating.values.astype("float32") / 5.0

        user_input = Input(shape=(1,))
        movie_input = Input(shape=(1,))
        user_emb = Embedding(len(user_ids), 8)(user_input)
        movie_emb = Embedding(len(movie_ids), 8)(movie_input)
        x = Concatenate()([Flatten()(user_emb), Flatten()(movie_emb)])
        x = Dense(16, activation="relu")(x)
        x = Dense(8, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        model = Model([user_input, movie_input], output)
        model.compile(optimizer="adam", loss="mse")
        model.fit([X_user, X_movie], y, epochs=80, batch_size=8, verbose=0)

        return model, u_map, m_map
    except Exception:
        return None, None, None

def neural_recommend(user_id, n=5):
    model, u_map, m_map = train_ncf()
    if model is None:
        return pd.Series(dtype=float)

    rated_ids = set(ratings[ratings.user_id == user_id].movie_id)
    candidates = [m for m in movies.movie_id if m not in rated_ids]
    X_u = np.array([u_map[user_id]] * len(candidates))
    X_m = np.array([m_map[m] for m in candidates])
    preds = model.predict([X_u, X_m], verbose=0).flatten()
    return pd.Series(preds, index=candidates).sort_values(ascending=False).head(n)

def hybrid_recommend(user_id, n=5):
    cf = collaborative_recommend(user_id, len(movies))
    cb = user_content_scores(user_id)

    cf_all = pd.Series(0.0, index=movies.movie_id)
    cf_all.loc[cf.index] = cf.values

    score = 0.6 * normalize_scores(cf_all) + 0.4 * normalize_scores(cb)

    rated_ids = set(ratings[ratings.user_id == user_id].movie_id)
    score.loc[list(rated_ids)] = -1
    return score.sort_values(ascending=False).head(n)

st.title("🎬 Smart Movie Recommendation System")
st.caption("CO1: Collaborative Filtering • CO2: TF-IDF Content Filtering • CO3: Hybrid • CO4: Neural Embeddings")

user_id = st.selectbox("Select User", sorted(ratings.user_id.unique()))
method = st.selectbox(
    "Recommendation Method",
    ["Hybrid", "Collaborative Filtering", "Content-Based", "Neural Collaborative Filtering"]
)

n = st.slider("Number of recommendations", 3, 8, 5)

st.subheader("⭐ Your Rated Movies")
rated = ratings[ratings.user_id == user_id].merge(movies, on="movie_id")
st.dataframe(rated[["title","genre","rating"]], use_container_width=True, hide_index=True)

if method == "Hybrid":
    result = hybrid_recommend(user_id, n)
elif method == "Collaborative Filtering":
    result = collaborative_recommend(user_id, n)
elif method == "Content-Based":
    result = user_content_scores(user_id)
    rated_ids = set(ratings[ratings.user_id == user_id].movie_id)
    result.loc[list(rated_ids)] = -1
    result = result.sort_values(ascending=False).head(n)
else:
    result = neural_recommend(user_id, n)

st.subheader(f"🎯 Recommended Movies — {method}")

if result.empty:
    st.warning("Neural model could not be loaded. Install TensorFlow using requirements.txt.")
else:
    out = movies[movies.movie_id.isin(result.index)].copy()
    out["score"] = out.movie_id.map(result)
    out = out.sort_values("score", ascending=False)
    st.dataframe(out[["title","genre","score"]], use_container_width=True, hide_index=True)

st.divider()
st.subheader("📌 How the system works")
st.markdown("""
- **CO1:** Item-based collaborative filtering + Matrix Factorization (SVD)
- **CO2:** TF-IDF + Cosine Similarity using movie descriptions
- **CO3:** Weighted Hybrid: **60% collaborative + 40% content**
- **CO4:** Neural Collaborative Filtering using **user and movie embeddings**
""")
