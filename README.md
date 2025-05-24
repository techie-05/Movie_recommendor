🎬 MovieLens Recommender System
A powerful, hybrid movie recommendation web app built with Streamlit, integrating collaborative filtering, content-based filtering, and sentiment analysis to help users discover movies they'll love — all wrapped in a sleek, dark-themed UI.

🚀 Features
Collaborative Filtering: Based on SVD matrix factorization and user-item interactions.

Content-Based Filtering: Uses genre similarity to recommend similar movies.

Sentiment Analysis: Predicts movie sentiments based on user-generated tags using Naive Bayes.

Hybrid Recommendations: Combines multiple approaches for high-quality suggestions.

Dark Theme UI: Custom CSS styling for a visually engaging experience.

Interactive Carousel Display: Scrollable and animated cards for movie recommendations.

Favorites & Search: Save favorite movies and search by title.

User Selection: Select from existing user IDs to simulate personalized recommendations.

🛠️ Tech Stack
Frontend: Streamlit

Backend/ML:

Scikit-learn (SVD, Naive Bayes, Cosine Similarity)

Pandas, NumPy

CountVectorizer (Text Processing)

🎯 How It Works
Recommendation Methods
Collaborative Filtering: Uses user-item rating matrix and SVD to infer hidden preferences.

Content Filtering: Averages genres from top-rated movies and finds similar genres.

Sentiment Model: Trained using user tag data with Naive Bayes classifier.

Hybrid Logic
Combines the outputs from both filtering methods and ranks them by sentiment confidence.

✨ Custom UI Design
Dark Mode enabled with elegant gradient buttons and cards.

Hover animations for movie cards.

Emoji-based sentiment indicators (😊, 😐) to enhance UX.

🙋‍♂️ Future Improvements
User authentication and login system

Allow adding/removing favorites dynamically

Real-time user feedback system

Integration with external movie APIs (e.g., TMDB for posters)
