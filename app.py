import os
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Basic Styling & Config
# =========================
st.set_page_config(
    page_title="MusicFinder AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .song-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Utilities
# =========================
def normalize_text(s: str) -> str:
    """Basic normalization."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_content_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    """Create a unified 'content' column for TF-IDF."""
    title_col = pick_first_existing_col(df, ["title", "track_name", "name", "song_title"])
    artist_col = pick_first_existing_col(df, ["artist", "artists", "artist_name", "singer"])
    text_col = pick_first_existing_col(df, ["lyrics_clean", "full", "lyrics", "text", "description", "lyric"])

    if title_col is None:
        title_col = "title"
        df[title_col] = "Unknown Song"
    if artist_col is None:
        artist_col = "artist"
        df[artist_col] = "Unknown Artist"

    if text_col is None:
        df["_text_raw"] = ""
        text_col = "_text_raw"

    for c in [title_col, artist_col, text_col]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    df["_title_norm"] = df[title_col].map(normalize_text)
    df["_artist_norm"] = df[artist_col].map(normalize_text)
    df["_text_norm"] = df[text_col].map(normalize_text)

    extra_cols = [c for c in ["emotion", "genre", "genres", "mood"] if c in df.columns]
    df["_extra_norm"] = ""
    for c in extra_cols:
        df["_extra_norm"] = df["_extra_norm"] + " " + df[c].fillna("").astype(str).map(normalize_text)
    df["_extra_norm"] = df["_extra_norm"].astype(str).str.strip()

    df["content"] = (
        df["_title_norm"] + " " +
        df["_artist_norm"] + " " +
        df["_text_norm"] +
        ((" " + df["_extra_norm"]) if extra_cols else "")
    ).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return df, title_col, artist_col, text_col

@st.cache_data
def load_csv(file_bytes: Optional[bytes], default_path: str) -> pd.DataFrame:
    if file_bytes is not None:
        try:
            return pd.read_csv(file_bytes)
        except:
            return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    
    # Create sample data
    sample_data = {
        'title': ['Hati-Hati di Jalan', 'Sang Dewi', 'Tak Ingin Usai', 'Berpisah Itu Mudah', 'Terlukis Indah'],
        'artist': ['Tulus', 'Lyodra', 'Keisya Levronka', 'Fiersa Besari', 'Rizky Febian'],
        'emotion': ['happy', 'romantic', 'sad', 'sad', 'romantic'],
        'genre': ['pop', 'pop', 'pop', 'acoustic', 'pop'],
        'lyrics': ['lirik lagu hati-hati di jalan', 'lirik lagu sang dewi', 'tak ingin usai', 'berpisah itu mudah', 'terlukis indah']
    }
    return pd.DataFrame(sample_data)

@st.cache_resource
def fit_vectorizer_and_matrix(content_series: pd.Series):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(content_series)
    sim = cosine_similarity(X, X)
    return vectorizer, sim

def recommend_by_index(df: pd.DataFrame, sim: np.ndarray, idx: int, top_n: int) -> pd.DataFrame:
    sims = list(enumerate(sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [x for x in sims if x[0] != idx]
    top = sims[:top_n]
    rec_idx = [i for i, _ in top]
    scores = [float(s) for _, s in top]
    out = df.iloc[rec_idx].copy()
    out["similarity"] = scores
    return out

def create_song_card(title, artist, similarity=None, emotion=None, genre=None):
    """Create a simple song card."""
    if similarity:
        similarity_text = f" ({similarity:.1%})"
    else:
        similarity_text = ""
    
    card = f"""
    <div class="song-card">
        <h4 style="margin: 0; color: #2d3748;">{title}{similarity_text}</h4>
        <p style="margin: 5px 0; color: #718096;">{artist}</p>
        <div style="display: flex; gap: 10px; margin-top: 8px;">
    """
    
    if emotion:
        card += f'<span style="background: #667eea20; color: #667eea; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{emotion}</span>'
    
    if genre:
        card += f'<span style="background: #4CAF8020; color: #4CAF80; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{genre}</span>'
    
    card += "</div></div>"
    return card

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🎵 MusicFinder AI")
    st.markdown("---")
    
    # File upload
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    
    # Navigation
    st.markdown("### Navigation")
    page = st.radio(
        "Select page:",
        ["🎯 Discover", "🔍 Search", "📊 Analytics"],
        label_visibility="collapsed"
    )
    
    # Load data
    try:
        df_raw = load_csv(uploaded, "indo-song-emot.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df_raw = load_csv(None, "sample")

# =========================
# Main Header
# =========================
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🎧 MusicFinder AI</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        AI-Powered Music Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Data Processing
# =========================
df = df_raw.copy()

if len(df) > 0:
    try:
        df, title_col, artist_col, text_col_used = build_content_column(df)
        df["_display"] = df[title_col].astype(str) + " • " + df[artist_col].astype(str)
        
        # Stats cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1.8rem;">{len(df):,}</h3>
                <p style="margin: 0; opacity: 0.9;">Total Songs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_artists = df[artist_col].nunique()
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4CAF80 0%, #2196F3 100%);">
                <h3 style="margin: 0; font-size: 1.8rem;">{unique_artists:,}</h3>
                <p style="margin: 0; opacity: 0.9;">Artists</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if "genre" in df.columns:
                unique_genres = df["genre"].nunique()
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #FF9800 0%, #FF5722 100%);">
                    <h3 style="margin: 0; font-size: 1.8rem;">{unique_genres}</h3>
                    <p style="margin: 0; opacity: 0.9;">Genres</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #FF9800 0%, #FF5722 100%);">
                    <h3 style="margin: 0; font-size: 1.8rem;">N/A</h3>
                    <p style="margin: 0; opacity: 0.9;">Genres</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

# =========================
# Main Content
# =========================
if page == "🎯 Discover":
    st.markdown("## 🎯 Discover Recommendations")
    
    if len(df) < 5:
        st.warning("Not enough songs in the dataset. Please upload a larger dataset.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Song selection
            search_term = st.text_input("Search songs...", placeholder="Type song or artist name")
            
            if search_term:
                search_term_lower = search_term.lower()
                filtered_songs = df[
                    df[title_col].str.lower().str.contains(search_term_lower, na=False) |
                    df[artist_col].str.lower().str.contains(search_term_lower, na=False)
                ]["_display"].tolist()
            else:
                filtered_songs = df["_display"].tolist()
            
            if filtered_songs:
                selected_song = st.selectbox("Choose a song", filtered_songs[:50])
            else:
                selected_song = None
            
            if selected_song:
                # Get song details
                song_idx = df.index[df["_display"] == selected_song][0]
                song_data = df.loc[song_idx]
                
                # Display selected song
                st.markdown("### Selected Song")
                st.markdown(create_song_card(
                    song_data[title_col],
                    song_data[artist_col],
                    emotion=song_data.get("emotion"),
                    genre=song_data.get("genre")
                ), unsafe_allow_html=True)
        
        with col2:
            # Settings
            top_n = st.slider("Number of recommendations", 5, 20, 10)
            
            # Filters
            if "emotion" in df.columns:
                emotions = ["All Emotions"] + sorted(df["emotion"].dropna().unique().tolist())
                selected_emotion = st.selectbox("Filter by emotion", emotions)
            else:
                selected_emotion = None
            
            if "genre" in df.columns:
                genres = ["All Genres"] + sorted(df["genre"].dropna().unique().tolist())
                selected_genre = st.selectbox("Filter by genre", genres)
            else:
                selected_genre = None
        
        # Generate recommendations
        if selected_song and st.button("Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding similar songs..."):
                # Filter dataset
                df_filtered = df.copy()
                if selected_emotion and selected_emotion != "All Emotions":
                    df_filtered = df_filtered[df_filtered["emotion"] == selected_emotion]
                if selected_genre and selected_genre != "All Genres":
                    df_filtered = df_filtered[df_filtered["genre"] == selected_genre]
                
                if len(df_filtered) < 2:
                    st.error("Not enough songs with the selected filters.")
                else:
                    # Fit model and get recommendations
                    vectorizer, sim = fit_vectorizer_and_matrix(df_filtered["content"])
                    
                    if selected_song in df_filtered["_display"].values:
                        idx = df_filtered.index[df_filtered["_display"] == selected_song][0]
                        pos = df_filtered.reset_index(drop=False).index[
                            df_filtered.reset_index(drop=False)["index"] == idx][0]
                        
                        recs = recommend_by_index(df_filtered.reset_index(drop=True), sim, pos, top_n)
                        
                        # Display results
                        st.markdown(f"### 🎵 Recommended Songs ({len(recs)} found)")
                        
                        for _, row in recs.iterrows():
                            st.markdown(create_song_card(
                                row[title_col],
                                row[artist_col],
                                similarity=row.get("similarity"),
                                emotion=row.get("emotion"),
                                genre=row.get("genre")
                            ), unsafe_allow_html=True)

elif page == "🔍 Search":
    st.markdown("## 🔍 Search Songs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search:",
            placeholder="Search by title, artist, or lyrics..."
        )
    
    with col2:
        search_limit = st.slider("Max results", 10, 50, 20)
    
    if search_query:
        # Simple search
        search_lower = search_query.lower()
        
        mask = (
            df[title_col].str.lower().str.contains(search_lower, na=False) |
            df[artist_col].str.lower().str.contains(search_lower, na=False)
        )
        
        if "content" in df.columns:
            mask = mask | df["content"].str.lower().str.contains(search_lower, na=False)
        
        if "emotion" in df.columns:
            mask = mask | df["emotion"].str.lower().str.contains(search_lower, na=False)
        
        if "genre" in df.columns:
            mask = mask | df["genre"].str.lower().str.contains(search_lower, na=False)
        
        filtered_df = df[mask]
        
        st.markdown(f"### Found {len(filtered_df)} songs")
        
        if not filtered_df.empty:
            # Display results
            for _, row in filtered_df.head(search_limit).iterrows():
                st.markdown(create_song_card(
                    row[title_col],
                    row[artist_col],
                    emotion=row.get("emotion"),
                    genre=row.get("genre")
                ), unsafe_allow_html=True)
        else:
            st.info("No songs found. Try a different search term.")

elif page == "📊 Analytics":
    st.markdown("## 📊 Dataset Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Song Distribution")
        
        # Artist distribution
        if artist_col in df.columns:
            top_artists = df[artist_col].value_counts().head(10)
            
            if len(top_artists) > 0:
                st.bar_chart(top_artists)
            else:
                st.info("Not enough artist data")
        
        # Word count if available
        if "content" in df.columns:
            df["word_count"] = df["content"].str.split().str.len()
            avg_words = df["word_count"].mean()
            st.metric("Average words per song", f"{avg_words:.0f}")
    
    with col2:
        st.markdown("### Dataset Info")
        
        # Basic info
        st.write(f"**Total songs:** {len(df)}")
        if artist_col in df.columns:
            st.write(f"**Unique artists:** {df[artist_col].nunique()}")
        
        if "emotion" in df.columns:
            emotion_counts = df["emotion"].value_counts()
            st.write("**Emotion distribution:**")
            for emotion, count in emotion_counts.items():
                st.write(f"  • {emotion}: {count}")
        
        if "genre" in df.columns:
            genre_counts = df["genre"].value_counts()
            st.write("**Genre distribution:**")
            for genre, count in genre_counts.head(5).items():
                st.write(f"  • {genre}: {count}")

# Footer
st.markdown("---")
st.markdown("🎵 **MusicFinder AI** • Powered by TF-IDF & Cosine Similarity")
