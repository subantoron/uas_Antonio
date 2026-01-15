import os
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Rekomendasi Musik (Content-Based • TF‑IDF + Cosine Similarity)",
    page_icon="🎵",
    layout="wide",
)

sns.set_style("whitegrid")

# =========================
# Utilities
# =========================
def normalize_text(s: str) -> str:
    """Basic Indonesian-friendly normalization."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)  # keep a-z and spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_content_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Create a unified 'content' column for TF-IDF from available columns.
    Returns: (df, title_col, artist_col, text_col_used)
    """
    title_col = pick_first_existing_col(df, ["title", "track_name", "name"])
    artist_col = pick_first_existing_col(df, ["artist", "artists", "artist_name"])
    text_col  = pick_first_existing_col(df, ["lyrics_clean", "full", "lyrics", "text", "description"])

    if title_col is None or artist_col is None:
        raise ValueError(
            "Kolom minimal tidak ditemukan. Dataset harus punya kolom judul & artist.\n"
            "Contoh: 'title' & 'artist' (seperti indo-song-emot.csv), atau 'name' & 'artists'."
        )

    # Create a clean lyrics/text field if not provided
    if text_col is None:
        df["_text_raw"] = ""
        text_col = "_text_raw"

    # Fill NA
    for c in [title_col, artist_col, text_col]:
        df[c] = df[c].fillna("").astype(str)

    # Normalize
    df["_title_norm"]  = df[title_col].map(normalize_text)
    df["_artist_norm"] = df[artist_col].map(normalize_text)
    df["_text_norm"]   = df[text_col].map(normalize_text)

    # Optional: emotion/genre can be included if present (row-wise)
    extra_cols = [c for c in ["emotion", "genre", "genres"] if c in df.columns]
    df["_extra_norm"] = ""
    for c in extra_cols:
        df["_extra_norm"] = df["_extra_norm"] + " " + df[c].fillna("").astype(str).map(normalize_text)
    df["_extra_norm"] = df["_extra_norm"].astype(str).str.strip()

    # Combine as content (ensure string, tidy spaces)
    df["content"] = (
        df["_title_norm"] + " " +
        df["_artist_norm"] + " " +
        df["_text_norm"] +
        ((" " + df["_extra_norm"]) if extra_cols else "")
    ).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return df, title_col, artist_col, text_col

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: Optional[bytes], default_path: str) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    raise FileNotFoundError(
        "Tidak ada file yang di-upload dan file default tidak ditemukan.\n"
        "Upload CSV kamu lewat sidebar."
    )

@st.cache_resource(show_spinner=False)
def fit_vectorizer_and_matrix(content_series: pd.Series) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit TF-IDF and return cosine similarity matrix (NxN).
    For large datasets, you'd want approximate nearest neighbors or row-wise similarity.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(content_series)
    sim = cosine_similarity(X, X)
    return vectorizer, sim

@st.cache_resource(show_spinner=False)
def fit_vectorizer_and_embeddings(content_series: pd.Series) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    """
    Fit TF-IDF and return embeddings matrix (sparse). Useful for query-based search.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(content_series)
    return vectorizer, X

def recommend_by_index(df: pd.DataFrame, sim: np.ndarray, idx: int, top_n: int) -> pd.DataFrame:
    sims = list(enumerate(sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [x for x in sims if x[0] != idx]  # drop itself
    top = sims[:top_n]
    rec_idx = [i for i, _ in top]
    scores = [float(s) for _, s in top]
    out = df.iloc[rec_idx].copy()
    out["similarity"] = scores
    return out

def recommend_by_query(df: pd.DataFrame, vectorizer: TfidfVectorizer, X, query: str, top_n: int) -> pd.DataFrame:
    q = normalize_text(query)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X).ravel()
    top_idx = np.argsort(-sims)[:top_n]
    out = df.iloc[top_idx].copy()
    out["similarity"] = [float(sims[i]) for i in top_idx]
    return out

# =========================
# Optional Spotify Lookup
# =========================
def _try_import_spotipy():
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        return spotipy, SpotifyClientCredentials
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def spotify_lookup_track(title: str, artist: str, client_id: str, client_secret: str):
    """
    Find track on Spotify (optional). Returns dict with spotify_url and preview_url.
    """
    spotipy, SpotifyClientCredentials = _try_import_spotipy()
    if spotipy is None:
        return {"spotify_url": None, "preview_url": None, "error": "spotipy belum terinstall."}

    try:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth)
        q = f"track:{title} artist:{artist}"
        res = sp.search(q=q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            return {"spotify_url": None, "preview_url": None, "error": "Tidak ditemukan di Spotify (search kosong)."}
        t = items[0]
        return {
            "spotify_url": (t.get("external_urls") or {}).get("spotify"),
            "preview_url": t.get("preview_url"),
            "error": None
        }
    except Exception as e:
        return {"spotify_url": None, "preview_url": None, "error": str(e)}

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Pengaturan")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

default_path = "indo-song-emot.csv"
df_raw = load_csv(uploaded.getvalue() if uploaded is not None else None, default_path=default_path)

st.sidebar.markdown("---")
st.sidebar.caption("Metode: Content-Based Filtering (TF‑IDF + Cosine Similarity)")

# Spotify integration (optional)
use_spotify = st.sidebar.checkbox("Opsional: cari & tampilkan link Spotify", value=False)
spotify_client_id = ""
spotify_client_secret = ""
if use_spotify:
    spotify_client_id = st.sidebar.text_input("SPOTIFY_CLIENT_ID", value=os.getenv("SPOTIFY_CLIENT_ID", ""), type="password")
    spotify_client_secret = st.sidebar.text_input("SPOTIFY_CLIENT_SECRET", value=os.getenv("SPOTIFY_CLIENT_SECRET", ""), type="password")
    if not spotify_client_id or not spotify_client_secret:
        st.sidebar.warning("Masukkan Client ID & Secret. Jika belum punya, buat di Spotify Developer Dashboard.")

# =========================
# Main: Data Prep
# =========================
st.title("🎵 Website Rekomendasi Musik ala Spotify")
st.write("Sistem rekomendasi **content-based** memakai **TF‑IDF** dan **cosine similarity**.")

with st.expander("📄 Lihat dataset (preview)"):
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.caption(f"Rows: {len(df_raw):,} • Columns: {len(df_raw.columns)}")

# Clean + build content
df = df_raw.copy()

# Drop duplicates (best effort)
if "title" in df.columns and "artist" in df.columns:
    df = df.drop_duplicates(subset=["title", "artist"], keep="first")
else:
    df = df.drop_duplicates()

df, title_col, artist_col, text_col_used = build_content_column(df)

# Provide ID for UI mapping
df["_display"] = df[title_col].astype(str) + " — " + df[artist_col].astype(str)

# Filter emotion if exists
emotion_filter = None
if "emotion" in df.columns:
    emotions = ["(Semua)"] + sorted([e for e in df["emotion"].dropna().astype(str).unique().tolist()])
    emotion_filter = st.sidebar.selectbox("Filter emotion (opsional)", emotions, index=0)

df_view = df
if emotion_filter and emotion_filter != "(Semua)":
    df_view = df[df["emotion"].astype(str) == emotion_filter].copy()

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Rekomendasi dari Lagu", "Rekomendasi dari Kata Kunci", "Visualisasi (EDA)"])

with tab1:
    st.subheader("🎧 Rekomendasi berdasarkan lagu yang kamu pilih")
    if len(df_view) < 5:
        st.error("Data terlalu sedikit setelah filter. Coba ganti filter.")
    else:
        # Fit similarity on df_view only (so filter affects results)
        vectorizer, sim = fit_vectorizer_and_matrix(df_view["content"])

        choice = st.selectbox("Pilih lagu", df_view["_display"].tolist())
        top_n = st.slider("Jumlah rekomendasi", min_value=3, max_value=30, value=10, step=1)

        idx = int(df_view.index[df_view["_display"] == choice][0])
        # df_view keeps original indices; map to position
        pos = df_view.reset_index(drop=False).index[df_view.reset_index(drop=False)["index"] == idx][0]
        recs = recommend_by_index(df_view.reset_index(drop=True), sim, pos, top_n)

        st.write("**Hasil rekomendasi:**")
        show_cols = [c for c in [title_col, artist_col, "emotion", "similarity"] if c in recs.columns]
        st.dataframe(recs[show_cols].sort_values("similarity", ascending=False), use_container_width=True)

        if use_spotify and spotify_client_id and spotify_client_secret:
            st.markdown("### 🔗 Link Spotify (opsional)")
            for _, row in recs.head(10).iterrows():
                title = str(row[title_col])
                artist = str(row[artist_col])
                info = spotify_lookup_track(title, artist, spotify_client_id, spotify_client_secret)
                colA, colB = st.columns([2, 3])
                with colA:
                    st.write(f"**{title}**")
                    st.caption(artist)
                    if "emotion" in recs.columns:
                        st.caption(f"emotion: {row.get('emotion')}")
                with colB:
                    if info.get("spotify_url"):
                        st.markdown(f"[Buka di Spotify]({info['spotify_url']})")
                    if info.get("preview_url"):
                        st.audio(info["preview_url"])
                    if info.get("error"):
                        st.caption(f"Catatan: {info['error']}")

with tab2:
    st.subheader("🔎 Rekomendasi berdasarkan kata kunci")
    st.caption("Contoh query: *'patah hati'*, *'bahagia'*, *'semangat'*, *'cinta'*, dll.")
    query = st.text_input("Masukkan kata kunci / deskripsi mood", value="patah hati")
    top_n = st.slider("Jumlah rekomendasi (query)", min_value=3, max_value=30, value=10, step=1, key="topn_query")

    vectorizer_q, X = fit_vectorizer_and_embeddings(df_view["content"])
    recs_q = recommend_by_query(df_view.reset_index(drop=True), vectorizer_q, X, query, top_n)

    show_cols = [c for c in [title_col, artist_col, "emotion", "similarity"] if c in recs_q.columns]
    st.dataframe(recs_q[show_cols].sort_values("similarity", ascending=False), use_container_width=True)

with tab3:
    st.subheader("📊 Visualisasi (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        # Type of visualization depends on available columns
        if "emotion" in df.columns:
            st.markdown("**Distribusi emotion**")
            fig = plt.figure(figsize=(6, 4))
            order = df["emotion"].astype(str).value_counts().index
            sns.countplot(y=df["emotion"].astype(str), order=order)
            plt.xlabel("Count")
            plt.ylabel("emotion")
            st.pyplot(fig)
        else:
            st.info("Kolom 'emotion' tidak ada di dataset ini.")

    with col2:
        # Top artists
        st.markdown("**Top 15 artist (frekuensi muncul)**")
        fig = plt.figure(figsize=(6, 4))
        top_art = df[artist_col].astype(str).value_counts().head(15).sort_values()
        plt.barh(top_art.index, top_art.values)
        plt.xlabel("Count")
        plt.ylabel("Artist")
        st.pyplot(fig)

    st.markdown("---")

    # Text length / word count if available
    if "word_count" in df.columns:
        st.markdown("**Distribusi panjang lirik (word_count)**")
        fig = plt.figure(figsize=(10, 4))
        plt.hist(pd.to_numeric(df["word_count"], errors="coerce").dropna(), bins=30)
        plt.xlabel("word_count")
        plt.ylabel("Jumlah lagu")
        st.pyplot(fig)
    else:
        st.caption("Tidak ada kolom 'word_count'. (Jika dataset kamu punya lirik, kamu bisa hitung word_count saat preprocessing.)")

st.markdown("---")
st.caption("© Content-Based Recommender • TF‑IDF + Cosine Similarity • Streamlit")
