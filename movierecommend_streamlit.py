import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Custom Dark Theme CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #181A20 !important;
        color: #ECECEC !important;
    }
    .stButton>button {
        background: linear-gradient(90deg,#23272F 70%,#1F222A 100%);
        color: #FFD700;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: #FF6F61;
        color: #fff;
        transform: scale(1.06);
    }
    .movie-card {
        background: #23272F;
        border-radius: 18px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.18);
        padding: 16px;
        margin: 16px 8px;
        transition: transform 0.2s;
        display: inline-block;
        width: 240px;
        vertical-align: top;
        cursor: pointer;
        border: 2px solid #2C2F36;
    }
    .movie-card:hover {
        transform: scale(1.04) rotateY(4deg);
        border: 2px solid #FFD700;
        background: linear-gradient(135deg, #23272F 80%, #222 100%);
    }
    .genre-chip {
        background: #31343B;
        border-radius: 8px;
        padding: 2px 8px;
        margin-right: 6px;
        font-size: 0.85em;
        color: #FFD700;
        display: inline-block;
    }
    .rating-badge {
        background: #FFD700;
        color: #181A20;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: bold;
        display: inline-block;
        margin-top: 8px;
    }
    .sentiment-pos { color: #00FF7F; font-size: 1.1em; }
    .sentiment-neg { color: #FF6F61; font-size: 1.1em; }
    .carousel-container {
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 8px;
    }
    .carousel-container::-webkit-scrollbar {
        height: 8px;
    }
    .carousel-container::-webkit-scrollbar-thumb {
        background: #23272F;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    return movies, ratings, tags

movies, ratings, tags = load_data()

# --- Preprocessing ---
data = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_movie_matrix)
reconstructed_ratings = np.dot(latent_matrix, svd.components_)
reconstructed_df = pd.DataFrame(reconstructed_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies['movieId'])
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = pd.merge(movies, movie_tags, on='movieId', how='left').fillna('')
movies_with_tags['sentiment'] = movies_with_tags['tag'].apply(
    lambda x: 1 if 'good' in x.lower() else (0 if 'bad' in x.lower() else 1)
)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(movies_with_tags['tag'])
y = movies_with_tags['sentiment']
clf = MultinomialNB()
clf.fit(X, y)

def collaborative_recommend(user_id, n=10):
    user_row = reconstructed_df.loc[user_id]
    user_rated = user_movie_matrix.loc[user_id]
    unseen = user_rated[user_rated == 0].index
    recs = user_row[unseen].sort_values(ascending=False).head(n)
    return recs.index.tolist()

def content_recommend(user_id, n=10):
    user_ratings = data[data['userId'] == user_id][['movieId', 'rating']]
    top_movies = user_ratings.sort_values(by='rating', ascending=False).head(5)['movieId']
    user_profile = genre_df.loc[top_movies].mean(axis=0).values.reshape(1, -1)
    similarities = cosine_similarity(user_profile, genre_df)[0]
    sim_scores = pd.Series(similarities, index=genre_df.index)
    already_rated = user_ratings['movieId'].tolist()
    recs = sim_scores.drop(already_rated).sort_values(ascending=False).head(n)
    return recs.index.tolist()

def sentiment_score(movie_ids):
    subset = movies_with_tags[movies_with_tags['movieId'].isin(movie_ids)]
    if subset.empty:
        return pd.Series(1, index=movie_ids)
    X_test = vectorizer.transform(subset['tag'])
    preds = clf.predict_proba(X_test)[:, 1]
    sentiment_scores = pd.Series(preds, index=subset['movieId'])
    return sentiment_scores.reindex(movie_ids, fill_value=sentiment_scores.mean())

def hybrid_recommend(user_id, n=5):
    collab_recs = collaborative_recommend(user_id, n=20)
    content_recs = content_recommend(user_id, n=20)
    combined = list(set(collab_recs) & set(content_recs))
    combined += [m for m in collab_recs + content_recs if m not in combined]
    combined = combined[:20]
    sentiments = sentiment_score(combined)
    top = sentiments.sort_values(ascending=False).head(n).index
    return movies[movies['movieId'].isin(top)]

# --- Helper functions for UI ---
def movie_card(movie, rating=None, sentiment=None):
    genres = movie['genres']
    genre_html = ''.join([f'<span class="genre-chip">{g}</span>' for g in genres])
    rating_html = f'<span class="rating-badge">{rating:.1f}</span>' if rating else ''
    sentiment_html = ''
    if sentiment is not None:
        if sentiment > 0.7:
            sentiment_html = '<span class="sentiment-pos">😊</span>'
        elif sentiment < 0.3:
            sentiment_html = '<span class="sentiment-neg">😐</span>'
        else:
            sentiment_html = '<span class="sentiment-neg">😐</span>'
    return f"""
    <div class="movie-card">
        <h4>{movie['title']}</h4>
        {genre_html}<br>
        {rating_html} {sentiment_html}
    </div>
    """

def carousel(movies_df, ratings=None, sentiments=None):
    html = '<div class="carousel-container">'
    for i, row in movies_df.iterrows():
        rating = ratings[row['movieId']] if ratings is not None and row['movieId'] in ratings else None
        sentiment = sentiments[row['movieId']] if sentiments is not None and row['movieId'] in sentiments else None
        html += movie_card(row, rating, sentiment)
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- Streamlit Pages ---
st.sidebar.title("🎬 MovieLens Recommender")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Recommendations", "⭐ Favorites", "🔍 Search", "👤 Profile"])

# --- Home Page ---
if page == "🏠 Home":
    st.markdown("<h1 style='color:#FFD700; font-size:3em;'>Welcome to MovieLens Recommender!</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1.2em; color:#ECECEC;'>
        <b>Discover movies you'll love.</b> <br>
        Get personalized recommendations powered by collaborative, content, and sentiment analysis.<br>
        <br>
        <span style='color:#FF6F61;'>Try the <b>Recommendations</b> tab to get started!</span>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=800&q=80", use_column_width=True)
    st.balloons()

# --- Recommendations Page ---
elif page == "🔮 Recommendations":
    st.markdown("<h2 style='color:#FFD700;'>Your Personalized Recommendations</h2>", unsafe_allow_html=True)
    user_id = st.selectbox("Select User ID", sorted(ratings['userId'].unique()), index=0)
    if st.button("Show Recommendations", key="recbtn"):
        recs = hybrid_recommend(user_id, n=5)
        sentiments = sentiment_score(recs['movieId'])
        carousel(recs, sentiments=sentiments)
        st.success("Scroll to see your top picks! Click 'Favorites' to save movies you like.")

# --- Favorites Page ---
elif page == "⭐ Favorites":
    st.markdown("<h2 style='color:#FFD700;'>Your Favorites</h2>", unsafe_allow_html=True)
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    fav_ids = st.session_state["favorites"]
    if fav_ids:
        fav_movies = movies[movies['movieId'].isin(fav_ids)]
        carousel(fav_movies)
    else:
        st.info("You have no favorites yet. Go to Recommendations and add some!")

# --- Search Page ---
elif page == "🔍 Search":
    st.markdown("<h2 style='color:#FFD700;'>Search for Movies</h2>", unsafe_allow_html=True)
    search = st.text_input("Type a movie title...")
    if search:
        results = movies[movies['title'].str.contains(search, case=False, na=False)].head(10)
        if not results.empty:
            carousel(results)
        else:
            st.warning("No movies found.")

# --- Profile Page ---
elif page == "👤 Profile":
    st.markdown("<h2 style='color:#FFD700;'>Your Profile</h2>", unsafe_allow_html=True)
    st.write("User settings and info coming soon!")
    st.progress(0.5)
    st.markdown("Dark theme enabled.")

# --- Footer Animation ---
st.markdown("""
    <hr style='border-color:#23272F;'>
    <div style='text-align:center; color:#ECECEC; font-size:0.9em;'>
        <span style='color:#FF6F61;'>Made with ❤ using Streamlit</span>
    </div>
""", unsafe_allow_html=True)