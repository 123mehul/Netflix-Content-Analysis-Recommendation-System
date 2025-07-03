import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from collections import Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NetflixContentAnalyzer:
    def __init__(self):
        # Power BI inspired color palette
        self.colors = {
            'primary': '#E50914',      # Netflix Red
            'secondary': '#221F1F',    # Netflix Dark
            'accent': '#F5F5F1',       # Netflix Light
            'success': '#46D369',      # Green
            'warning': '#FFB800',      # Orange
            'info': '#0073E6',         # Blue
            'purple': '#8B5CF6',       # Purple
            'pink': '#EC4899',         # Pink
            'background': '#F8F9FA',
            'text': '#323130',
            'grid': '#E1E1E1'
        }
        
        # Set professional styling
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette([self.colors['primary'], self.colors['info'], self.colors['success'], self.colors['warning']])
        
        # Initialize ML components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.content_matrix = None
        self.df = None
    
    def load_netflix_data(self):
        """Load Netflix dataset or generate realistic mock data"""
        print("📺 Loading Netflix content data...")
        
        try:
            # Try to load real Netflix dataset
            # You can download from: https://www.kaggle.com/shivamb/netflix-shows
            self.df = pd.read_csv('netflix_titles.csv')
            print(f"✅ Loaded real Netflix dataset with {len(self.df)} titles")
        except FileNotFoundError:
            print("📊 Generating comprehensive mock Netflix dataset...")
            self.df = self.generate_mock_netflix_data()
            print(f"✅ Generated mock dataset with {len(self.df)} titles")
        
        return self.preprocess_data()
    
    def generate_mock_netflix_data(self):
        """Generate realistic Netflix content data"""
        np.random.seed(42)
        
        # Content types and genres
        content_types = ['Movie', 'TV Show']
        genres = [
            'Action & Adventure', 'Comedies', 'Dramas', 'Horror Movies', 'Thrillers',
            'Documentaries', 'Romantic Movies', 'Sci-Fi & Fantasy', 'Crime TV Shows',
            'Kids & Family Movies', 'International Movies', 'Anime Features',
            'Stand-Up Comedy', 'Music & Musicals', 'Sports Movies', 'Classic Movies'
        ]
        
        countries = [
            'United States', 'India', 'United Kingdom', 'Canada', 'France',
            'Germany', 'Japan', 'South Korea', 'Spain', 'Italy', 'Brazil',
            'Mexico', 'Australia', 'Netherlands', 'Turkey', 'Argentina'
        ]
        
        ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA']
        
        directors = [
            'Christopher Nolan', 'Martin Scorsese', 'Quentin Tarantino', 'Steven Spielberg',
            'David Fincher', 'Ridley Scott', 'Denis Villeneuve', 'Jordan Peele',
            'Greta Gerwig', 'Rian Johnson', 'Chloe Zhao', 'Barry Jenkins'
        ]
        
        # Generate data
        data = []
        for i in range(8000):  # Generate 8000 titles
            # Random release year (Netflix content from 1925 to 2024)
            release_year = np.random.randint(1925, 2025)
            
            # Date added (mostly recent years)
            if release_year >= 2015:
                date_added = f"{np.random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} {np.random.randint(1, 29)}, {np.random.randint(2016, 2025)}"
            else:
                date_added = f"{np.random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} {np.random.randint(1, 29)}, {np.random.randint(2008, 2025)}"
            
            content_type = np.random.choice(content_types, p=[0.7, 0.3])  # More movies than TV shows
            
            # Generate title based on genre
            genre_list = np.random.choice(genres, size=np.random.randint(1, 4), replace=False)
            listed_in = ', '.join(genre_list)
            
            # Generate realistic titles
            title_templates = [
                "The {adjective} {noun}",
                "{noun} of {place}",
                "The Last {noun}",
                "{adjective} {noun}: The {sequel}",
                "Beyond the {noun}",
                "{place}: A {noun} Story",
                "The {adjective} {profession}",
                "{noun} Chronicles"
            ]
            
            adjectives = ['Dark', 'Silent', 'Hidden', 'Lost', 'Secret', 'Final', 'Golden', 'Broken', 'Wild', 'Perfect']
            nouns = ['Journey', 'Mystery', 'Adventure', 'Story', 'Legend', 'Dream', 'Shadow', 'Light', 'Heart', 'Soul']
            places = ['Tokyo', 'Paris', 'London', 'Mumbai', 'Seoul', 'Berlin', 'Madrid', 'Rome', 'Sydney', 'Toronto']
            professions = ['Detective', 'Artist', 'Chef', 'Teacher', 'Doctor', 'Lawyer', 'Writer', 'Musician']
            sequels = ['Beginning', 'Return', 'Awakening', 'Revolution', 'Legacy', 'Origins', 'Destiny']
            
            template = np.random.choice(title_templates)
            title = template.format(
                adjective=np.random.choice(adjectives),
                noun=np.random.choice(nouns),
                place=np.random.choice(places),
                profession=np.random.choice(professions),
                sequel=np.random.choice(sequels)
            )
            
            # Duration based on type
            if content_type == 'Movie':
                duration = f"{np.random.randint(80, 180)} min"
            else:
                seasons = np.random.randint(1, 8)
                duration = f"{seasons} Season{'s' if seasons > 1 else ''}"
            
            # Description
            descriptions = [
                f"A {genre_list[0].lower()} story about love, loss, and redemption.",
                f"An epic {genre_list[0].lower()} that follows the journey of unlikely heroes.",
                f"A gripping {genre_list[0].lower()} that explores the depths of human nature.",
                f"A heartwarming {genre_list[0].lower()} about family, friendship, and finding your place in the world.",
                f"A thrilling {genre_list[0].lower()} that will keep you on the edge of your seat.",
                f"An inspiring {genre_list[0].lower()} based on true events that changed history.",
                f"A mind-bending {genre_list[0].lower()} that challenges everything you think you know.",
                f"A beautiful {genre_list[0].lower()} that celebrates the power of hope and perseverance."
            ]
            
            data.append({
                'show_id': f's{i+1}',
                'type': content_type,
                'title': title,
                'director': np.random.choice(directors + ['Unknown Director']),
                'cast': ', '.join(np.random.choice(['Actor A', 'Actor B', 'Actor C', 'Actor D', 'Actor E'], 
                                                 size=np.random.randint(1, 6), replace=False)),
                'country': np.random.choice(countries),
                'date_added': date_added,
                'release_year': release_year,
                'rating': np.random.choice(ratings),
                'duration': duration,
                'listed_in': listed_in,
                'description': np.random.choice(descriptions)
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self):
        """Clean and preprocess the Netflix data"""
        print("🧹 Preprocessing data...")
        
        # Convert date_added to datetime
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        
        # Extract year and month from date_added
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        
        # Clean duration column
        self.df['duration_value'] = self.df['duration'].str.extract('(\d+)').astype(float)
        self.df['duration_type'] = self.df['duration'].str.extract('(min|Season)')
        
        # Split genres
        self.df['genres_list'] = self.df['listed_in'].str.split(', ')
        
        # Create content features for ML
        self.df['content_features'] = (
            self.df['listed_in'].fillna('') + ' ' + 
            self.df['description'].fillna('') + ' ' + 
            self.df['cast'].fillna('') + ' ' + 
            self.df['director'].fillna('')
        )
        
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=['title', 'type', 'release_year'])
        
        print(f"✅ Preprocessed {len(self.df)} titles")
        return self.df
    
    def build_recommendation_engine(self):
        """Build content-based recommendation engine"""
        print("🤖 Building recommendation engine...")
        
        # Create TF-IDF matrix
        self.content_matrix = self.tfidf_vectorizer.fit_transform(self.df['content_features'])
        
        print(f"✅ Built recommendation engine with {self.content_matrix.shape[1]} features")
    
    def get_recommendations(self, title, num_recommendations=10):
        """Get content-based recommendations"""
        try:
            # Find the index of the title
            idx = self.df[self.df['title'].str.contains(title, case=False, na=False)].index[0]
            
            # Calculate cosine similarity
            sim_scores = cosine_similarity(self.content_matrix[idx:idx+1], self.content_matrix).flatten()
            
            # Get top similar titles
            sim_indices = sim_scores.argsort()[::-1][1:num_recommendations+1]
            
            recommendations = self.df.iloc[sim_indices][['title', 'type', 'listed_in', 'release_year', 'rating']]
            recommendations['similarity_score'] = sim_scores[sim_indices]
            
            return recommendations
        except IndexError:
            return pd.DataFrame()
    
    def create_executive_dashboard(self):
        """Create comprehensive Netflix analytics dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Content Distribution', 'Content Added Over Time', 'Top Genres',
                'Country-wise Content', 'Release Year Distribution', 'Rating Distribution',
                'Content Duration Analysis', 'Monthly Addition Trends', 'Key Statistics'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # 1. Content Distribution (Movies vs TV Shows)
        content_counts = self.df['type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=content_counts.index,
                values=content_counts.values,
                hole=0.4,
                marker_colors=[self.colors['primary'], self.colors['info']],
                textinfo='label+percent+value',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Content Added Over Time
        yearly_additions = self.df.groupby('year_added').size().reset_index()
        yearly_additions.columns = ['year', 'count']
        yearly_additions = yearly_additions.dropna()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_additions['year'],
                y=yearly_additions['count'],
                mode='lines+markers',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=8, color=self.colors['primary']),
                name='Content Added',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Top Genres
        all_genres = []
        for genres_list in self.df['genres_list'].dropna():
            all_genres.extend(genres_list)
        
        genre_counts = Counter(all_genres).most_common(10)
        genres, counts = zip(*genre_counts)
        
        fig.add_trace(
            go.Bar(
                x=list(counts),
                y=list(genres),
                orientation='h',
                marker_color=self.colors['success'],
                name='Genre Count',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Country-wise Content
        top_countries = self.df['country'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=top_countries.index,
                y=top_countries.values,
                marker_color=self.colors['warning'],
                name='Country Content',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Release Year Distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['release_year'],
                nbinsx=30,
                marker_color=self.colors['purple'],
                name='Release Years',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. Rating Distribution
        rating_counts = self.df['rating'].value_counts().head(8)
        fig.add_trace(
            go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker_color=self.colors['pink'],
                name='Ratings',
                showlegend=False
            ),
            row=2, col=3
        )
        
        # 7. Content Duration Analysis
        movies_duration = self.df[self.df['type'] == 'Movie']['duration_value'].dropna()
        tv_duration = self.df[self.df['type'] == 'TV Show']['duration_value'].dropna()
        
        fig.add_trace(
            go.Box(
                y=movies_duration,
                name='Movies (min)',
                marker_color=self.colors['primary'],
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Box(
                y=tv_duration,
                name='TV Shows (seasons)',
                marker_color=self.colors['info'],
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 8. Monthly Addition Trends
        monthly_additions = self.df.groupby('month_added').size().reset_index()
        monthly_additions.columns = ['month', 'count']
        monthly_additions = monthly_additions.dropna()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_additions['month_name'] = monthly_additions['month'].map(
            {i+1: month_names[i] for i in range(12)}
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_additions['month_name'],
                y=monthly_additions['count'],
                marker_color=self.colors['success'],
                name='Monthly Additions',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # 9. Key Statistics Table
        total_content = len(self.df)
        total_movies = len(self.df[self.df['type'] == 'Movie'])
        total_tv_shows = len(self.df[self.df['type'] == 'TV Show'])
        total_countries = self.df['country'].nunique()
        avg_movie_duration = self.df[self.df['type'] == 'Movie']['duration_value'].mean()
        latest_year = self.df['release_year'].max()
        oldest_year = self.df['release_year'].min()
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Content', f"{total_content:,}"],
            ['Movies', f"{total_movies:,}"],
            ['TV Shows', f"{total_tv_shows:,}"],
            ['Countries', f"{total_countries}"],
            ['Avg Movie Duration', f"{avg_movie_duration:.0f} min"],
            ['Latest Release', f"{latest_year}"],
            ['Oldest Content', f"{oldest_year}"],
            ['Top Genre', f"{genres[0]}"],
            ['Top Country', f"{top_countries.index[0]}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*stats_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🎬 Netflix Content Analytics Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10, color=self.colors['text']),
            height=1200,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update axes styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        
        fig.show()
    
    def create_genre_analysis_dashboard(self):
        """Create detailed genre analysis dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Genre Popularity Over Time',
                'Genre vs Content Type',
                'Genre Rating Distribution',
                'Genre Duration Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "violin"}]
            ]
        )
        
        # 1. Genre Popularity Over Time
        # Get top 5 genres
        all_genres = []
        for genres_list in self.df['genres_list'].dropna():
            all_genres.extend(genres_list)
        top_5_genres = [genre for genre, _ in Counter(all_genres).most_common(5)]
        
        for i, genre in enumerate(top_5_genres):
            genre_by_year = []
            for year in range(2015, 2025):
                year_data = self.df[self.df['year_added'] == year]
                genre_count = sum(1 for genres_list in year_data['genres_list'].dropna() 
                                if genre in genres_list)
                genre_by_year.append(genre_count)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(2015, 2025)),
                    y=genre_by_year,
                    mode='lines+markers',
                    name=genre,
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Genre vs Content Type
        genre_type_data = []
        for genre in top_5_genres:
            movies = sum(1 for idx, row in self.df.iterrows() 
                        if row['type'] == 'Movie' and genre in (row['genres_list'] or []))
            tv_shows = sum(1 for idx, row in self.df.iterrows() 
                          if row['type'] == 'TV Show' and genre in (row['genres_list'] or []))
            genre_type_data.append({'genre': genre, 'movies': movies, 'tv_shows': tv_shows})
        
        genre_df = pd.DataFrame(genre_type_data)
        
        fig.add_trace(
            go.Bar(
                x=genre_df['genre'],
                y=genre_df['movies'],
                name='Movies',
                marker_color=self.colors['primary']
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=genre_df['genre'],
                y=genre_df['tv_shows'],
                name='TV Shows',
                marker_color=self.colors['info']
            ),
            row=1, col=2
        )
        
        # 3. Genre Rating Distribution
        for genre in top_5_genres[:3]:  # Top 3 for clarity
            genre_ratings = []
            for idx, row in self.df.iterrows():
                if genre in (row['genres_list'] or []):
                    genre_ratings.append(row['rating'])
            
            rating_counts = Counter(genre_ratings)
            
            fig.add_trace(
                go.Box(
                    y=list(rating_counts.keys()),
                    name=genre
                ),
                row=2, col=1
            )
        
        # 4. Genre Duration Analysis
        for genre in top_5_genres[:3]:
            genre_durations = []
            for idx, row in self.df.iterrows():
                if (genre in (row['genres_list'] or []) and 
                    row['type'] == 'Movie' and 
                    pd.notna(row['duration_value'])):
                    genre_durations.append(row['duration_value'])
            
            if genre_durations:
                fig.add_trace(
                    go.Violin(
                        y=genre_durations,
                        name=genre,
                        box_visible=True
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title={
                'text': '🎭 Netflix Genre Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800
        )
        
        fig.show()
    
    def create_recommendation_dashboard(self, sample_title=None):
        """Create recommendation engine dashboard"""
        if sample_title is None:
            # Get a random popular title
            sample_title = self.df.sample(1)['title'].iloc[0]
        
        recommendations = self.get_recommendations(sample_title, 8)
        
        if recommendations.empty:
            print(f"❌ No recommendations found for '{sample_title}'")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Recommendations for "{sample_title}"',
                'Similarity Scores',
                'Recommended Content Types',
                'Recommendation Engine Stats'
            ),
            specs=[
                [{"type": "table"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "table"}]
            ]
        )
        
        # 1. Recommendations Table
        rec_data = [
            ['Title', 'Type', 'Genre', 'Year', 'Similarity'],
            *[[row['title'][:30] + '...' if len(row['title']) > 30 else row['title'],
               row['type'], 
               row['listed_in'][:20] + '...' if len(row['listed_in']) > 20 else row['listed_in'],
               str(row['release_year']), 
               f"{row['similarity_score']:.3f}"] 
              for _, row in recommendations.iterrows()]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=rec_data[0],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=11)),
                cells=dict(values=list(zip(*rec_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=10))
            ),
            row=1, col=1
        )
        
        # 2. Similarity Scores
        fig.add_trace(
            go.Bar(
                x=recommendations['similarity_score'],
                y=recommendations['title'].str[:20] + '...',
                orientation='h',
                marker_color=self.colors['success'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Recommended Content Types
        type_counts = recommendations['type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.4,
                marker_colors=[self.colors['primary'], self.colors['info']],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Engine Stats
        total_features = self.content_matrix.shape[1]
        avg_similarity = recommendations['similarity_score'].mean()
        max_similarity = recommendations['similarity_score'].max()
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Features', f"{total_features:,}"],
            ['Recommendations', f"{len(recommendations)}"],
            ['Avg Similarity', f"{avg_similarity:.3f}"],
            ['Max Similarity', f"{max_similarity:.3f}"],
            ['Content Types', f"{recommendations['type'].nunique()}"],
            ['Unique Genres', f"{recommendations['listed_in'].nunique()}"],
            ['Year Range', f"{recommendations['release_year'].min()}-{recommendations['release_year'].max()}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.colors['info'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*stats_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '🤖 Netflix Recommendation Engine Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800
        )
        
        fig.show()
        
        return recommendations
    
    def create_content_clustering_analysis(self):
        """Create content clustering analysis using ML"""
        print("🔬 Performing content clustering analysis...")
        
        # Reduce dimensionality for clustering
        pca = PCA(n_components=50)
        content_pca = pca.fit_transform(self.content_matrix.toarray())
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=8, random_state=42)
        clusters = kmeans.fit_predict(content_pca)
        
        # Add cluster labels to dataframe
        self.df['cluster'] = clusters
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Content Clusters (PCA Visualization)',
                'Cluster Distribution',
                'Cluster Characteristics',
                'Cluster Genre Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "bar"}]
            ]
        )
        
        # 1. PCA Visualization
        # Further reduce to 2D for visualization
        pca_2d = PCA(n_components=2)
        content_2d = pca_2d.fit_transform(content_pca)
        
        colors_list = px.colors.qualitative.Set3
        for cluster_id in range(8):
            cluster_data = content_2d[clusters == cluster_id]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data[:, 0],
                    y=cluster_data[:, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(color=colors_list[cluster_id], size=6, opacity=0.7)
                ),
                row=1, col=1
            )
        
        # 2. Cluster Distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                marker_color=colors_list[:len(cluster_counts)],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Cluster Characteristics
        cluster_chars = []
        for cluster_id in range(8):
            cluster_df = self.df[self.df['cluster'] == cluster_id]
            avg_year = cluster_df['release_year'].mean()
            top_type = cluster_df['type'].mode().iloc[0] if not cluster_df['type'].mode().empty else 'Unknown'
            top_country = cluster_df['country'].mode().iloc[0] if not cluster_df['country'].mode().empty else 'Unknown'
            
            cluster_chars.append([
                f'Cluster {cluster_id}',
                f"{len(cluster_df)}",
                f"{avg_year:.0f}",
                top_type,
                top_country[:15]
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Cluster', 'Size', 'Avg Year', 'Top Type', 'Top Country'],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=11)),
                cells=dict(values=list(zip(*cluster_chars)),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=10))
            ),
            row=2, col=1
        )
        
        # 4. Cluster Genre Analysis
        cluster_genre_data = []
        for cluster_id in range(min(4, 8)):  # Show top 4 clusters
            cluster_df = self.df[self.df['cluster'] == cluster_id]
            all_genres = []
            for genres_list in cluster_df['genres_list'].dropna():
                all_genres.extend(genres_list)
            
            if all_genres:
                top_genre = Counter(all_genres).most_common(1)[0]
                cluster_genre_data.append({
                    'cluster': f'Cluster {cluster_id}',
                    'top_genre': top_genre[0],
                    'count': top_genre[1]
                })
        
        if cluster_genre_data:
            cluster_genre_df = pd.DataFrame(cluster_genre_data)
            fig.add_trace(
                go.Bar(
                    x=cluster_genre_df['cluster'],
                    y=cluster_genre_df['count'],
                    text=cluster_genre_df['top_genre'],
                    textposition='auto',
                    marker_color=colors_list[:len(cluster_genre_df)],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title={
                'text': '🔬 Netflix Content Clustering Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800
        )
        
        fig.show()
    
    def display_summary_report(self):
        """Display comprehensive text summary"""
        print(f"\n🎬 NETFLIX CONTENT ANALYSIS REPORT")
        print("=" * 60)
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Content Overview
        total_content = len(self.df)
        movies = len(self.df[self.df['type'] == 'Movie'])
        tv_shows = len(self.df[self.df['type'] == 'TV Show'])
        
        print(f"\n📊 CONTENT OVERVIEW")
        print("-" * 30)
        print(f"🎭 Total Content: {total_content:,}")
        print(f"🎬 Movies: {movies:,} ({movies/total_content*100:.1f}%)")
        print(f"📺 TV Shows: {tv_shows:,} ({tv_shows/total_content*100:.1f}%)")
        print(f"🌍 Countries: {self.df['country'].nunique()}")
        print(f"📅 Year Range: {self.df['release_year'].min()} - {self.df['release_year'].max()}")
        
        # Genre Analysis
        all_genres = []
        for genres_list in self.df['genres_list'].dropna():
            all_genres.extend(genres_list)
        
        top_genres = Counter(all_genres).most_common(5)
        
        print(f"\n🎭 TOP GENRES")
        print("-" * 20)
        for i, (genre, count) in enumerate(top_genres, 1):
            print(f"{i}. {genre}: {count:,} titles")
        
        # Country Analysis
        top_countries = self.df['country'].value_counts().head(5)
        
        print(f"\n🌍 TOP COUNTRIES")
        print("-" * 25)
        for i, (country, count) in enumerate(top_countries.items(), 1):
            print(f"{i}. {country}: {count:,} titles")
        
        # Content Trends
        recent_years = self.df[self.df['year_added'] >= 2020]['year_added'].value_counts().sort_index()
        
        print(f"\n📈 RECENT TRENDS (2020+)")
        print("-" * 30)
        for year, count in recent_years.items():
            print(f"📅 {year}: {count:,} titles added")
        
        # Duration Analysis
        avg_movie_duration = self.df[self.df['type'] == 'Movie']['duration_value'].mean()
        avg_tv_seasons = self.df[self.df['type'] == 'TV Show']['duration_value'].mean()
        
        print(f"\n⏱️ DURATION INSIGHTS")
        print("-" * 25)
        print(f"🎬 Average Movie Duration: {avg_movie_duration:.0f} minutes")
        print(f"📺 Average TV Show Seasons: {avg_tv_seasons:.1f}")
        
        # Rating Analysis
        top_ratings = self.df['rating'].value_counts().head(3)
        
        print(f"\n🏷️ TOP RATINGS")
        print("-" * 20)
        for rating, count in top_ratings.items():
            print(f"📋 {rating}: {count:,} titles")
        
        print(f"\n✅ Analysis complete! Check your browser for interactive dashboards.")

def main():
    analyzer = NetflixContentAnalyzer()
    
    print("🎬 Netflix Content Analysis & Recommender")
    print("💼 Power BI-Style Visualizations & ML")
    print("=" * 50)
    
    # Load and preprocess data
    analyzer.load_netflix_data()
    
    # Build recommendation engine
    analyzer.build_recommendation_engine()
    
    print("\nSelect analysis type:")
    print("1. Complete dashboard suite (All analyses)")
    print("2. Executive overview only")
    print("3. Genre analysis")
    print("4. Recommendation engine demo")
    print("5. Content clustering analysis")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        # Complete analysis
        print("\n🎨 Creating comprehensive Netflix dashboards...")
        analyzer.create_executive_dashboard()
        analyzer.create_genre_analysis_dashboard()
        
        # Demo recommendation engine
        sample_titles = analyzer.df.sample(3)['title'].tolist()
        for title in sample_titles:
            print(f"\n🤖 Generating recommendations for: {title}")
            analyzer.create_recommendation_dashboard(title)
        
        analyzer.create_content_clustering_analysis()
        analyzer.display_summary_report()
        
    elif choice == '2':
        analyzer.create_executive_dashboard()
        analyzer.display_summary_report()
        
    elif choice == '3':
        analyzer.create_genre_analysis_dashboard()
        
    elif choice == '4':
        title = input("Enter a title to get recommendations (or press Enter for random): ").strip()
        if not title:
            title = analyzer.df.sample(1)['title'].iloc[0]
        
        recommendations = analyzer.create_recommendation_dashboard(title)
        
        if not recommendations.empty:
            print(f"\n🎯 TOP RECOMMENDATIONS FOR '{title}':")
            print("-" * 50)
            for i, (_, rec) in enumerate(recommendations.head(5).iterrows(), 1):
                print(f"{i}. {rec['title']} ({rec['type']}, {rec['release_year']})")
                print(f"   Genre: {rec['listed_in']}")
                print(f"   Similarity: {rec['similarity_score']:.3f}")
                print()
        
    elif choice == '5':
        analyzer.create_content_clustering_analysis()
        
    else:
        analyzer.create_executive_dashboard()
    
    print(f"\n✅ Netflix analysis complete!")

if __name__ == "__main__":
    main()  