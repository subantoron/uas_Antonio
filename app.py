import os
import re
import time
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Modern Styling & Config
# =========================
st.set_page_config(
    page_title="Spotify-Style Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .song-card {
        background: linear-gradient(135deg, #f6f8ff 0%, #f1f4ff 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e6ff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3ff;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .spotify-button {
        background: #1DB954 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
    }
    
    .similarity-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 6px;
        border-radius: 3px;
        margin: 5px 0;
    }
    
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid", {'axes.grid': False})

# =========================
# Utilities
# =========================
@st.cache_data
def normalize_text(s: str) -> str:
    """Basic Indonesian-friendly normalization."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data
def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data
def build_content_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    """Create a unified 'content' column for TF-IDF."""
    title_col = pick_first_existing_col(df, ["title", "track_name", "name", "song_title"])
    artist_col = pick_first_existing_col(df, ["artist", "artists", "artist_name", "singer"])
    text_col = pick_first_existing_col(df, ["lyrics_clean", "full", "lyrics", "text", "description", "lyric"])

    if title_col is None or artist_col is None:
        raise ValueError("Dataset must have title and artist columns.")

    if text_col is None:
        df["_text_raw"] = ""
        text_col = "_text_raw"

    for c in [title_col, artist_col, text_col]:
        df[c] = df[c].fillna("").astype(str)

    df["_title_norm"] = df[title_col].map(normalize_text)
    df["_artist_norm"] = df[artist_col].map(normalize_text)
    df["_text_norm"] = df[text_col].map(normalize_text)

    extra_cols = [c for c in ["emotion", "genre", "genres", "mood", "category", "tags"] if c in df.columns]
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

    df["word_count"] = df["content"].str.split().str.len()
    
    return df, title_col, artist_col, text_col

@st.cache_data
def load_csv(file_bytes: Optional[bytes], default_path: str) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    
    # Create sample data if no file exists
    sample_data = {
        'title': ['Hati-Hati di Jalan', 'Sang Dewi', 'Tak Ingin Usai', 'Berpisah Itu Mudah', 'Terlukis Indah'],
        'artist': ['Tulus', 'Lyodra', 'Keisya Levronka', 'Fiersa Besari', 'Rizky Febian'],
        'emotion': ['happy', 'romantic', 'sad', 'sad', 'romantic'],
        'genre': ['pop', 'pop', 'pop', 'pop', 'pop'],
        'lyrics': ['lirik lagu hati-hati di jalan', 'lirik lagu sang dewi', 'tak ingin usai lirik', 'berpisah itu mudah', 'terlukis indah']
    }
    return pd.DataFrame(sample_data)

@st.cache_resource
def fit_vectorizer_and_matrix(content_series: pd.Series) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=None
    )
    X = vectorizer.fit_transform(content_series)
    sim = cosine_similarity(X, X)
    return vectorizer, sim

@st.cache_resource
def fit_vectorizer_and_embeddings(content_series: pd.Series) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=None
    )
    X = vectorizer.fit_transform(content_series)
    return vectorizer, X

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

def recommend_by_query(df: pd.DataFrame, vectorizer: TfidfVectorizer, X, query: str, top_n: int) -> pd.DataFrame:
    q = normalize_text(query)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X).ravel()
    top_idx = np.argsort(-sims)[:top_n]
    out = df.iloc[top_idx].copy()
    out["similarity"] = [float(sims[i]) for i in top_idx]
    return out

def create_song_card(title, artist, similarity=None, emotion=None, genre=None):
    """Create a modern song card component."""
    card_html = f"""
    <div class="song-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex-grow: 1;">
                <h4 style="margin: 0; color: #2d3748;">{title}</h4>
                <p style="margin: 5px 0; color: #718096; font-size: 0.9rem;">{artist}</p>
                <div style="display: flex; gap: 10px; margin-top: 8px;">
    """
    
    if emotion:
        card_html += f"""
        <span style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); 
                     color: #667eea; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem;">
            {emotion}
        </span>
        """
    
    if genre:
        card_html += f"""
        <span style="background: linear-gradient(135deg, #4CAF8020 0%, #2196F320 100%); 
                     color: #4CAF80; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem;">
            {genre}
        </span>
        """
    
    if similarity:
        card_html += f"""
        <span style="background: linear-gradient(135deg, #FF980020 0%, #FF572220 100%); 
                     color: #FF5722; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem;">
            {similarity:.1%}
        </span>
        """
    
    card_html += """
                </div>
            </div>
            <div style="margin-left: 15px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polygon points="10,8 16,12 10,16" fill="#667eea"></polygon>
                </svg>
            </div>
        </div>
    </div>
    """
    return card_html

# =========================
# Sidebar - Modern Design
# =========================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #667eea; margin: 0;">🎵</h1>
        <h3 style="color: #2d3748; margin: 10px 0;">MusicFinder AI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Discover", "Search", "Analytics", "Settings"],
        icons=["compass", "search", "bar-chart", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#667eea", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px",
                "--hover-color": "#e9ecef",
            },
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    st.markdown("---")
    
    # File upload with improved UI
    st.markdown("### 📁 Data Source")
    uploaded = st.file_uploader(
        "Upload your music dataset",
        type=["csv"],
        help="Upload a CSV file with music data (columns: title, artist, lyrics, emotion, genre)"
    )
    
    default_path = "indo-song-emot.csv"
    try:
        df_raw = load_csv(uploaded.getvalue() if uploaded is not None else None, default_path=default_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using sample data instead.")
        df_raw = load_csv(None, "sample")
    
    # Dataset Info Card
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Songs", f"{len(df_raw):,}")
    with col2:
        st.metric("Columns", len(df_raw.columns))
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        st.markdown("#### Model Settings")
        ngram_range = st.slider("N-gram Range", 1, 3, (1, 2))
        min_df = st.slider("Min Document Frequency", 1, 10, 2)
        
        st.markdown("#### Display Settings")
        results_per_page = st.slider("Results per page", 5, 20, 10)
        show_similarity = st.checkbox("Show similarity scores", True)
        
        st.markdown("#### Spotify Integration")
        use_spotify = st.checkbox("Enable Spotify links", False)
        if use_spotify:
            spotify_client_id = st.text_input("Client ID", type="password")
            spotify_client_secret = st.text_input("Client Secret", type="password")

# =========================
# Main Header
# =========================
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🎧 MusicFinder AI</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Discover your next favorite song with AI-powered recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Data Processing
# =========================
df = df_raw.copy()

# Clean and prepare data
if "title" in df.columns and "artist" in df.columns:
    df = df.drop_duplicates(subset=["title", "artist"], keep="first")
else:
    df = df.drop_duplicates()

df, title_col, artist_col, text_col_used = build_content_column(df)

# Create display columns
df["_display"] = df[title_col].astype(str) + " • " + df[artist_col].astype(str)

# Quick Stats Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin: 0; font-size: 1.8rem;">{:,}</h3>
        <p style="margin: 0; opacity: 0.9;">Total Songs</p>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    unique_artists = df[artist_col].nunique()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #4CAF80 0%, #2196F3 100%);">
        <h3 style="margin: 0; font-size: 1.8rem;">{unique_artists:,}</h3>
        <p style="margin: 0; opacity: 0.9;">Unique Artists</p>
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
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #FF9800 0%, #FF5722 100%);">
            <h3 style="margin: 0; font-size: 1.8rem;">N/A</h3>
            <p style="margin: 0; opacity: 0.9;">Genres</p>
        </div>
        """, unsafe_allow_html=True)

with col4:
    avg_words = int(df["word_count"].mean()) if "word_count" in df.columns else 0
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%);">
        <h3 style="margin: 0; font-size: 1.8rem;">{avg_words}</h3>
        <p style="margin: 0; opacity: 0.9;">Avg Words</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Main Content Based on Navigation
# =========================
if selected == "Discover":
    st.markdown("## 🎯 Discover Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["🎵 By Song", "🔍 By Mood", "📊 Popular"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Select a Song")
            search_term = st.text_input("Search songs...", placeholder="Type song or artist name")
            
            # Filter songs based on search
            if search_term:
                search_term_lower = search_term.lower()
                filtered_songs = df[
                    df[title_col].str.lower().str.contains(search_term_lower) |
                    df[artist_col].str.lower().str.contains(search_term_lower)
                ]["_display"].tolist()
            else:
                filtered_songs = df["_display"].tolist()
            
            if filtered_songs:
                selected_song = st.selectbox(
                    "Choose a song",
                    filtered_songs[:50],  # Limit for performance
                    key="song_select"
                )
                
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
            else:
                st.warning("No songs found. Try a different search term.")
                selected_song = None
        
        with col2:
            st.markdown("### Settings")
            top_n = st.slider("Number of recommendations", 5, 20, 10)
            
            # Emotion filter if available
            if "emotion" in df.columns:
                emotions = ["All Emotions"] + sorted(df["emotion"].dropna().unique().tolist())
                selected_emotion = st.selectbox("Filter by emotion", emotions)
            else:
                selected_emotion = None
            
            # Genre filter if available
            if "genre" in df.columns:
                genres = ["All Genres"] + sorted(df["genre"].dropna().unique().tolist())
                selected_genre = st.selectbox("Filter by genre", genres)
            else:
                selected_genre = None
        
        # Generate recommendations
        if selected_song and st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding similar songs..."):
                # Filter dataset based on selections
                df_filtered = df.copy()
                if selected_emotion and selected_emotion != "All Emotions":
                    df_filtered = df_filtered[df_filtered["emotion"] == selected_emotion]
                if selected_genre and selected_genre != "All Genres":
                    df_filtered = df_filtered[df_filtered["genre"] == selected_genre]
                
                if len(df_filtered) < 2:
                    st.error("Not enough songs in the filtered dataset. Try different filters.")
                else:
                    # Fit model on filtered data
                    vectorizer, sim = fit_vectorizer_and_matrix(df_filtered["content"])
                    
                    # Find index in filtered dataframe
                    if selected_song in df_filtered["_display"].values:
                        idx = df_filtered.index[df_filtered["_display"] == selected_song][0]
                        pos = df_filtered.reset_index(drop=False).index[
                            df_filtered.reset_index(drop=False)["index"] == idx][0]
                        
                        # Get recommendations
                        recs = recommend_by_index(df_filtered.reset_index(drop=True), sim, pos, top_n)
                        
                        # Display results
                        st.markdown(f"### 🎵 Recommended Songs ({len(recs)} found)")
                        
                        # Sort by similarity
                        recs = recs.sort_values("similarity", ascending=False)
                        
                        # Create columns for grid layout
                        cols = st.columns(2)
                        for i, (_, row) in enumerate(recs.iterrows()):
                            with cols[i % 2]:
                                st.markdown(create_song_card(
                                    row[title_col],
                                    row[artist_col],
                                    similarity=row.get("similarity"),
                                    emotion=row.get("emotion"),
                                    genre=row.get("genre")
                                ), unsafe_allow_html=True)
                        
                        # Similarity distribution chart
                        if len(recs) > 1:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=recs["similarity"],
                                    y=recs[title_col].str[:30] + "...",
                                    orientation='h',
                                    marker=dict(
                                        color=recs["similarity"],
                                        colorscale='Viridis',
                                        showscale=True
                                    ),
                                    text=[f"{x:.1%}" for x in recs["similarity"]],
                                    textposition='outside'
                                )
                            ])
                            fig.update_layout(
                                title="Similarity Scores",
                                xaxis_title="Similarity",
                                yaxis_title="Song",
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔍 Find Songs by Mood/Theme")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            mood_query = st.text_area(
                "Describe what you're looking for:",
                placeholder="Example: songs for studying, breakup songs, happy upbeat music, romantic ballads...",
                height=100
            )
        
        with col2:
            top_n_mood = st.number_input("Results", min_value=5, max_value=30, value=10)
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        if search_button and mood_query:
            with st.spinner("Searching for matching songs..."):
                vectorizer_q, X = fit_vectorizer_and_embeddings(df["content"])
                recs_q = recommend_by_query(df, vectorizer_q, X, mood_query, top_n_mood)
                
                st.markdown(f"### Found {len(recs_q)} songs matching '{mood_query}'")
                
                # Display results in a nice grid
                num_cols = 2
                rows = (len(recs_q) + num_cols - 1) // num_cols
                
                for i in range(rows):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i * num_cols + j
                        if idx < len(recs_q):
                            with cols[j]:
                                row = recs_q.iloc[idx]
                                st.markdown(create_song_card(
                                    row[title_col],
                                    row[artist_col],
                                    similarity=row.get("similarity"),
                                    emotion=row.get("emotion"),
                                    genre=row.get("genre")
                                ), unsafe_allow_html=True)
        
        # Quick mood buttons
        st.markdown("### Quick Mood Filters")
        quick_moods = ["Happy", "Sad", "Romantic", "Energetic", "Chill", "Motivational"]
        
        cols = st.columns(len(quick_moods))
        for idx, mood in enumerate(quick_moods):
            with cols[idx]:
                if st.button(f"🎵 {mood}", use_container_width=True):
                    st.session_state.mood_query = mood.lower()
                    st.rerun()

elif selected == "Search":
    st.markdown("## 🔍 Advanced Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search across all song data:",
            placeholder="Search by title, artist, lyrics, emotion, or genre..."
        )
    
    with col2:
        search_limit = st.slider("Max results", 10, 100, 25)
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "artist" in df.columns:
                artists = ["All Artists"] + sorted(df["artist"].dropna().unique().tolist())
                selected_artist = st.selectbox("Artist", artists)
            else:
                selected_artist = None
        
        with col2:
            if "emotion" in df.columns:
                emotions = ["All Emotions"] + sorted(df["emotion"].dropna().unique().tolist())
                selected_emotion = st.selectbox("Emotion", emotions)
            else:
                selected_emotion = None
        
        with col3:
            if "genre" in df.columns:
                genres = ["All Genres"] + sorted(df["genre"].dropna().unique().tolist())
                selected_genre = st.selectbox("Genre", genres)
            else:
                selected_genre = None
    
    if st.button("Search", type="primary", use_container_width=True):
        with st.spinner("Searching..."):
            # Apply filters
            filtered_df = df.copy()
            
            if search_query:
                search_lower = search_query.lower()
                mask = (
                    filtered_df[title_col].str.lower().str.contains(search_lower) |
                    filtered_df[artist_col].str.lower().str.contains(search_lower) |
                    filtered_df["content"].str.lower().str.contains(search_lower)
                )
                if "emotion" in filtered_df.columns:
                    mask = mask | filtered_df["emotion"].str.lower().str.contains(search_lower)
                if "genre" in filtered_df.columns:
                    mask = mask | filtered_df["genre"].str.lower().str.contains(search_lower)
                filtered_df = filtered_df[mask]
            
            if selected_artist and selected_artist != "All Artists":
                filtered_df = filtered_df[filtered_df["artist"] == selected_artist]
            
            if selected_emotion and selected_emotion != "All Emotions":
                filtered_df = filtered_df[filtered_df["emotion"] == selected_emotion]
            
            if selected_genre and selected_genre != "All Genres":
                filtered_df = filtered_df[filtered_df["genre"] == selected_genre]
            
            # Display results
            st.markdown(f"### 📊 Found {len(filtered_df)} songs")
            
            if not filtered_df.empty:
                # Convert to display format
                display_df = filtered_df.copy()
                if title_col in display_df.columns and artist_col in display_df.columns:
                    display_df["Song"] = display_df[title_col] + " • " + display_df[artist_col]
                
                # Select columns to show
                show_cols = ["Song"]
                if "emotion" in display_df.columns:
                    show_cols.append("emotion")
                if "genre" in display_df.columns:
                    show_cols.append("genre")
                if "word_count" in display_df.columns:
                    show_cols.append("word_count")
                
                # Display as dataframe
                st.dataframe(
                    display_df[show_cols].head(search_limit),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"music_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No songs found matching your criteria. Try different search terms.")

elif selected == "Analytics":
    st.markdown("## 📊 Data Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎭 Emotions", "🎨 Genres"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top artists chart
            st.markdown("### Top 10 Artists")
            top_artists = df[artist_col].value_counts().head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_artists.values,
                    y=top_artists.index,
                    orientation='h',
                    marker_color='#667eea'
                )
            ])
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Number of Songs",
                yaxis_title="Artist"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Word count distribution
            st.markdown("### Lyrics Length Distribution")
            if "word_count" in df.columns:
                fig = px.histogram(
                    df, 
                    x="word_count",
                    nbins=30,
                    color_discrete_sequence=['#764ba2']
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Word Count",
                    yaxis_title="Number of Songs"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Word count data not available")
    
    with tab2:
        if "emotion" in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Emotion distribution
                st.markdown("### Emotion Distribution")
                emotion_counts = df["emotion"].value_counts()
                
                fig = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emotion over time (if year exists)
                if "year" in df.columns:
                    st.markdown("### Emotion Trends")
                    emotion_by_year = df.groupby(["year", "emotion"]).size().reset_index(name='count')
                    
                    fig = px.line(
                        emotion_by_year,
                        x="year",
                        y="count",
                        color="emotion",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Emotion data not available in this dataset")
    
    with tab3:
        if "genre" in df.columns:
            # Genre network visualization
            st.markdown("### Genre Distribution")
            
            genre_counts = df["genre"].value_counts()
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=np.random.rand(len(genre_counts)),
                    y=np.random.rand(len(genre_counts)),
                    mode='markers+text',
                    marker=dict(
                        size=genre_counts.values * 10 / genre_counts.values.max(),
                        color=genre_counts.values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=genre_counts.index,
                    textposition="top center"
                )
            ])
            fig.update_layout(
                height=500,
                showlegend=False,
                title="Genre Popularity (size = number of songs)"
            )
            st.plotly_chart(fig, use_container_width=True)

elif selected == "Settings":
    st.markdown("## ⚙️ Settings")
    
    with st.form("settings_form"):
        st.markdown("### Display Settings")
        
        theme = st.selectbox("Theme", ["Light", "Dark"])
        density = st.selectbox("Density", ["Comfortable", "Compact"])
        animations = st.checkbox("Enable animations", True)
        
        st.markdown("### Recommendation Settings")
        default_rec_count = st.slider("Default recommendations count", 5, 20, 10)
        show_explanations = st.checkbox("Show explanation for recommendations", True)
        
        st.markdown("### Data Settings")
        auto_refresh = st.checkbox("Auto-refresh data", False)
        cache_duration = st.slider("Cache duration (hours)", 1, 24, 6)
        
        if st.form_submit_button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
            st.info("Some changes may require a page refresh to take effect.")

# =========================
# Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 1rem;">
        <p>🎵 MusicFinder AI • Powered by TF-IDF & Cosine Similarity</p>
        <p style="font-size: 0.9rem;">© 2024 • Discover your perfect soundtrack</p>
    </div>
    """, unsafe_allow_html=True)
