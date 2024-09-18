import re
import string
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

REQS = [
    ('pip', 'pip==24.2'),
    ('matplotlib', 'matplotlib==3.9.2'),
    ('nltk', 'nltk==3.9.1'),
    ('numpy', 'numpy==2.1.1'),
    ('optuna', 'optuna==4.0.0'),
    ('pandas', 'pandas==2.2.2'),
    ('seaborn', 'seaborn==0.13.2'),
    ('sklearn', 'scikit-learn==1.5.2'),
    ('statsmodels', 'statsmodels==0.14.3')
    ('umap', 'umap==0.1.1')
]

try:
    subprocess.check_call([sys.executable, '-m', 'ensurepip'])
except Exception as e:
    print(e, file=sys.stderr)


def ensure_installed(module_info):
    _, install_str = module_info
    try:
        subprocess.check_call([sys.executable, '-m',
                               'pip', 'install', '--quiet',
                               install_str])
        print(f'Installed "{install_str}".')
    except Exception as e:
        print(e, file=sys.stderr)


for m in REQS:
    ensure_installed(m)

# Standard libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Machine learning and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mean_squared_error

# Statistical modeling
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Natural Language Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Dimensionality reduction
import umap

# Hyperparameter optimization
import optuna


def find_columns_with_missing(data, columns):
    """Finding features that have a lot of missing data"""
    print()
    print('Finding columns with missing data...')
    missing = []
    i = 0
    for col in columns:
        missing.append(data[col].isnull().sum())
        if missing[i] > 0:
            print()
            print(f'Column {col} is missing {missing[i]} values.')
            print(f'Proportion of missing data is {missing[i]/len(data)}.')
            if missing[i]/len(data) >= 0.9:
                print(f'Dropping column {col}...')
                data = data.drop(columns=col)
                data_cleaned = data
        i += 1
    return missing, data_cleaned


def hex_to_rgb(hex_color):
    """Function to convert hex to RGB"""
    # Remove the '#' if it exists
    hex_color = hex_color.lstrip('#')

    # Convert hex to integer and split into RGB components
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]


def preprocess_text(text):
    """Preprocessing function"""
    text = text.lower()
    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removes punctuation
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    return ' '.join(tokens)


def plot_silhouette_bar_across_experiments(model_names, silhouette_scores):
    n_experiments = len(silhouette_scores)
    n_models = len(model_names)
    bar_width = 0.2
    index = np.arange(n_experiments)
    plt.figure(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        sil_scores = [exp_scores[i] for exp_scores in silhouette_scores]
        plt.bar(index + i * bar_width,sil_scores, bar_width, label=model_name)

    plt.xlabel('Experiments')
    plt.ylabel('Silhouette scores')
    plt.title('Silhouette scores Across Models and Experiments')
    plt.xticks(index + bar_width * (n_models - 1) / 2, [f'Exp {i+1}' for i in range(n_experiments)])
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_ch_index_across_experiments(model_names, ch_scores):

    n_experiments = len(ch_scores)
    n_models = len(model_names)
    bar_width = 0.2
    index = np.arange(n_experiments)
    plt.figure(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        ch_score = [exp_scores[i] for exp_scores in ch_scores]
        plt.bar(index + i * bar_width, ch_score, bar_width, label=model_name)

    plt.xlabel('Experiments')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title('Calinski-Harabasz Index Across Models and Experiments')
    plt.xticks(index + bar_width * (n_models - 1) / 2, [f'Exp {i+1}' for i in range(n_experiments)])
    plt.legend()
    plt.tight_layout()
    plt.show()


class KMeansClustering:
    def __init__(self, data):
        self.data = data
        self.best_params = None
        self.kmeans_model = None

    def tune_hyperparameters(self, n_trials=50):
        def objective_kmeans(trial):
            n_clusters = trial.suggest_int('n_clusters', 2, 10)
            init_method = trial.suggest_categorical('init', ['k-means++', 'random'])

            kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=42)
            kmeans.fit(self.data)
            labels = kmeans.labels_
            score = silhouette_score(self.data, labels)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_kmeans, n_trials=n_trials)
        self.best_params = study.best_params
        print("Best params:", self.best_params)

    def fit_model(self):
        self.kmeans_model = KMeans(n_clusters=self.best_params['n_clusters'],
                                   init=self.best_params['init'],
                                   random_state=42)
        self.kmeans_model.fit(self.data)

    def visualize_clusters(self, umap_embedding, feature):
        labels = self.kmeans_model.labels_
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot in 3D
        scatter = ax.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            umap_embedding[:, 2],
            c=labels,
            cmap='viridis',
            s=30
        )
        # Add labels and title
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')
        plt.title(f'3D UMAP of K-Means Clusters on {feature}')
        # Add a color bar for better visual distinction of clusters
        plt.colorbar(scatter)
        # Show the plot
        plt.show()

    def plot_elbow_method(self, k_range=(2, 10)):
        """
        Plot the Elbow Method for choosing the optimal number of clusters
        Args:
        - k_range: tuple, range of cluster numbers to evaluate
        """
        inertia = []
        K = range(k_range[0], k_range[1] + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center

        plt.figure(figsize=(8, 6))
        plt.plot(K, inertia, 'bo-', markersize=8)
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia (Sum of squared distances)')
        plt.grid(True)
        plt.show()

    def output_label(self):
        return self.kmeans_model.labels_

    def silhoutte(self):
        score = silhouette_score(self.data, self.kmeans_model.labels_)
        print(f'The Silhouette score is {score}')
        return score

    def calinski(self):
        if len(np.unique(self.kmeans_model.labels_)) > 1:  # Only calculate if there are clusters
            score = calinski_harabasz_score(self.data, self.kmeans_model.labels_)
        else:
            score = np.nan  # If only one cluster (or all noise), set to NaN
        print(f'The Callinski index is {score}')
        return score


class DBSCANClustering:
    def __init__(self, data):
        self.data = data
        self.best_params = None
        self.dbscan_model = None

    def tune_hyperparameters(self, n_trials=50):
        def objective_dbscan(trial):
            eps = trial.suggest_float('eps', 0.1, 2.0)
            min_samples = trial.suggest_int('min_samples', 3, 20)

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(self.data)
            labels = dbscan.labels_
            if len(set(labels)) > 1:
                score = silhouette_score(self.data, labels)
            else:
                score = -1
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_dbscan, n_trials=n_trials)
        self.best_params = study.best_params
        print("Found best params:", self.best_params)

    def fit_model(self):
        self.dbscan_model = DBSCAN(eps=self.best_params['eps'], min_samples=self.best_params['min_samples'])
        self.dbscan_model.fit(self.data)

    def visualize_clusters_and_outliers_3D(self, umap_embedding, feature):
        labels = self.dbscan_model.labels_

        # Separate clustered points and noise points
        clustered_points = umap_embedding[labels >= 0]  # Points part of a cluster
        clustered_labels = labels[labels >= 0]
        outliers = umap_embedding[labels == -1]  # Noise points

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the clustered points in different colors
        scatter = ax.scatter(clustered_points[:, 0], clustered_points[:, 1], clustered_points[:, 2],
                             c=clustered_labels, cmap='viridis', s=30)

        # Plot the outliers (noise points) in red with 'x' markers
        ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='red', marker='x', s=80, label='Outliers')

        # Add labels and title
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')
        ax.set_title(f'DBSCAN 3D Clusters with Outliers {feature}')
        # Add a legend and color bar for clusters
        plt.legend()
        plt.colorbar(scatter, ax=ax)
        plt.show()

    def output_label(self):
        return self.dbscan_model.labels_

    def silhoutte(self):
        score = silhouette_score(self.data, self.dbscan_model.labels_)
        print(f'The Silhouette score is {score}')
        return score

    def calinski(self):
        if len(np.unique(self.dbscan_model.labels_)) > 1:  # Only calculate if there are clusters
            score = calinski_harabasz_score(self.data, self.dbscan_model.labels_)
        else:
            score = np.nan  # If only one cluster (or all noise), set to NaN
        print(f'The Callinski index is {score}')
        return score


class ClusteringDataRetriever:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def get_data_with_labels(self):
        # If Data is in a numpy array, convert it to a pandas DataFrame
        if isinstance(self.data, np.ndarray):
            df = pd.DataFrame(self.data)
        else:
            df = self.data.copy()  # If already a DataFrame

        # Add a new column for the cluster labels
        df['Cluster_Label'] = self.labels

        return df[['gender', 'gender:confidence', 'Cluster_Label']]

    def get_cluster_data(self, cluster_label):
        # Retrieve data points belonging to a specific cluster.
        df = self.get_data_with_labels()
        return df[df['Cluster_Label'] == cluster_label]

    def get_noise_data(self):
        # Retrieve Data points classified as noise (-1 label) in DBSCAN.
        return self.get_cluster_data(-1)


# Main starts here
# Load the dataset
df = pd.read_csv('twitter_user_data.csv', encoding='ISO-8859-1')

# Quick view of the dataset
print()
print('Dataset Overview')
print(df.info())
print(df.head())

all_features = df.columns

missing_col, df_cleaned = find_columns_with_missing(df, all_features)

# Dropping rows where 'gender' is missing
df_cleaned = df_cleaned.dropna(subset=['gender'])

# Drop the 'profile_yn' column since it is not relevant to human/non-human classification
df_cleaned = df_cleaned.drop(columns=['profile_yn'])

# Now that we have handled the missing data, you can proceed with further analysis
print()
print('Dataset Overview')
print(df_cleaned.info())
print(df_cleaned.head())

print()
print('---- EXPLORATORY DATA ANALYSIS (EDA) ----')

current_num_features = df.select_dtypes(include=[np.number])

# Plot distribution of each numerical feature with gender as hue using seaborn
for feature in current_num_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df_cleaned, x=feature, hue='gender', bins=30, kde=True)
    plt.title(f'Distribution of {feature} by Gender')
    plt.show()

# Distribution of gender
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df_cleaned)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('count')
plt.show()

# Plot distribution of 'tweet_count' and 'retweet_count'
for column in ['tweet_count', 'retweet_count']:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_cleaned, x=column, kde=True, bins=30)
    plt.title(f'Distribution of {column.replace("_", " ").capitalize()}')
    plt.show()

# Correlation analysis for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned[['tweet_count', 'retweet_count', 'fav_number']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Extracting date from 'created' and 'tweet_created' for time-based analysis
df_cleaned['profile_created_year'] = pd.to_datetime(df_cleaned['created']).dt.year
df_cleaned['tweet_created_year'] = pd.to_datetime(df_cleaned['tweet_created']).dt.year

# Ensure 'created' and tweet_created are in datetime format
df_cleaned['created'] = pd.to_datetime(df_cleaned['created'], errors='coerce')
df_cleaned['tweet_created'] = pd.to_datetime(df_cleaned['tweet_created'], errors='coerce')

# assuming Data was up-to-date
df_cleaned['account_age'] = (pd.Timestamp.now() - df_cleaned['created']).dt.days

df_cleaned['tweets_per_day'] = df_cleaned['tweet_count'] / df_cleaned['account_age']
df_cleaned['retweets_per_day'] = df_cleaned['retweet_count'] / df_cleaned['account_age']
df_cleaned['favorites_per_day'] = df_cleaned['fav_number'] / df_cleaned['account_age']

# Plotting the distribution of profile creation over the years
plt.figure(figsize=(8, 6))
sns.histplot(df_cleaned['profile_created_year'], kde=False, bins=15)
plt.title('Distribution of Profile Creation Years')
plt.xlabel('Profile Created Year')
plt.ylabel('count')
plt.show()

# Plotting the histogram of tweets per day
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['tweets_per_day'], bins=50, kde=True)
plt.title('Distribution of Tweets Per Day')
plt.xlabel('Tweets Per Day')
plt.ylabel('Frequency')
plt.show()

# show the relationship between account age and tweets per day
plt.figure(figsize=(10, 6))
sns.scatterplot(x='account_age', y='tweets_per_day', data=df_cleaned)
plt.title('Account Age vs. Tweets Per Day')
plt.xlabel('Account Age (Days)')
plt.ylabel('Tweets Per Day')
plt.show()

# Exploring 'link_color' and 'sidebar_color' features

# Check number of NaN value in  'link_color' and 'sidebar_color' features
link_color_nan_count = df_cleaned['link_color'].isnull().sum()
sidebar_color_nan_count = df_cleaned['sidebar_color'].isnull().sum()

print()
print(f"Number of NaN values in 'link_color': {link_color_nan_count}.")
print(f"Number of NaN values in 'sidebar_color': {sidebar_color_nan_count}.")

# Check how many available colors in 'link_color' and 'sidebar_color' features
link_color_count = len(df_cleaned['link_color'].unique())
sidebar_color_count = len(df_cleaned['sidebar_color'].unique())
print(f'Number of link color is {link_color_count}.')
print(f'Number of side bar color is {sidebar_color_count}.')

# Apply the function to 'link_color' and 'sidebar_color'
df_cleaned['link_color'] = df_cleaned['link_color'].apply(lambda x: f'#{x}' if len(x) == 6 else '#000000')
df_cleaned['sidebar_color'] = df_cleaned['sidebar_color'].apply(lambda x: f'#{x}' if len(x) == 6 else '#000000')

# Drop rows where 'sidebar_color' is still NaN
df_cleaned = df_cleaned.dropna(subset=['link_color'])
df_cleaned = df_cleaned.dropna(subset=['sidebar_color'])
print(f"Number of NaN values in 'link_color': {df_cleaned['link_color'].isnull().sum()}")
print(f"Number of NaN values in 'sidebar_color': {df_cleaned['sidebar_color'].isnull().sum()}")

# top 15 colors
top_sidebar_colors = df_cleaned['sidebar_color'].value_counts().iloc[:15].index.tolist()
top_link_colors = df_cleaned['link_color'].value_counts().iloc[:15].index.tolist()
# print(top_sidebar_colors)

# Extract top 10 most common sidebar colors
sns.set(rc={'axes.facecolor':'lightgrey', 'figure.facecolor':'white'})
plt.figure(figsize=(8, 6))
sns.countplot(y='sidebar_color', data=df_cleaned, order=df_cleaned['sidebar_color'].value_counts().iloc[:15].index, palette=top_sidebar_colors)
plt.title('Top 15 Most Common Profile sidebar_color')
plt.ylabel('Sidebar Color')
plt.xlabel('count')
plt.grid()
plt.show()

# Extract top 10 most common link colors
sns.set(rc={'axes.facecolor':'lightgrey', 'figure.facecolor':'white'})
plt.figure(figsize=(8, 6))
sns.countplot(y='link_color', data=df_cleaned, order=df_cleaned['link_color'].value_counts().iloc[:15].index, palette=top_link_colors)
plt.title('Top 15 Most Common Profile link_color')
plt.ylabel('Link Color')
plt.xlabel('count')
plt.grid()
plt.show()

# count plot for sidebar_color vs. gender
plt.figure(figsize=(10, 6))
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.countplot(x='sidebar_color', hue='gender', data=df_cleaned,
              order=df_cleaned['sidebar_color'].value_counts().iloc[:15].index)
plt.title('Top 15 Most Common Sidebar Colors by Gender')
plt.xlabel('Sidebar Color')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.show()

# count plot for link_color vs. gender
plt.figure(figsize=(10, 6))
sns.countplot(x='link_color', hue='gender', data=df_cleaned,
              order=df_cleaned['link_color'].value_counts().iloc[:15].index)
plt.title('Top 15 Most Common link Colors by Gender')
plt.xlabel('Link Color')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.show()

# Scatter plot for link_color vs. tweet_count with gender as hue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='link_color', y='tweet_count', hue='gender', data=df_cleaned[df_cleaned['link_color'].isin(top_link_colors)],
                palette='Set2', s=100, alpha=0.7)
plt.title('Link Colors vs. Tweet count with Gender')
plt.xlabel('Link Color')
plt.ylabel('Tweet count')
plt.xticks(rotation=45)
plt.show()

# Scatter plot for sidebar_color vs. tweet_count with gender as hue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sidebar_color', y='tweet_count', hue='gender', data=df_cleaned[df_cleaned['sidebar_color'].isin(top_sidebar_colors)],
                palette='Set2', s=100, alpha=0.7)
plt.title('Sidebar Colors vs. Tweet count with Gender')
plt.xlabel('Sidebar Color')
plt.ylabel('Tweet count')
plt.xticks(rotation=45)
plt.show()

# Select columns to be used
col = ['gender', 'gender:confidence', 'description', 'favorites_per_day','link_color',
       'retweets_per_day', 'sidebar_color', 'text', 'tweets_per_day','user_timezone', 'tweet_location', 'profile_created_year', 'tweet_created_year'
       ]
df_preprocessed = df_cleaned[col].copy()
# Remove rows where gender is 'Unknown'
df_preprocessed = df_preprocessed[df_preprocessed['gender'] != 'unknown']

# Plot correlation matrix
corr_matrix = df_preprocessed.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Drop one feature from highly correlated pairs (correlation > 0.9)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
df_preprocessed = df_preprocessed.drop(columns=to_drop)

# Filling missing values for important features
df_preprocessed['user_timezone'].fillna('Unknown', inplace=True)
df_preprocessed['tweet_location'].fillna('Unknown', inplace=True)
categorical_features = ['user_timezone', 'tweet_location']

# categorise types of features

# numerical features
df_num = df_preprocessed[['retweets_per_day', 'favorites_per_day', 'tweets_per_day', 'profile_created_year', 'tweet_created_year']].copy()

# categorical features with frequency encoding
freq_encoding_location = df_preprocessed['tweet_location'].value_counts(normalize=True)
df_preprocessed['tweet_location_encoded'] = df_preprocessed['tweet_location'].map(freq_encoding_location)

freq_encoding_timezone = df_preprocessed['user_timezone'].value_counts(normalize=True)
df_preprocessed['user_timezone_encoded'] = df_preprocessed['user_timezone'].map(freq_encoding_timezone)

# gender features
# encode the 'gender' column to numeric values
df_preprocessed['gender'] = df_preprocessed['gender'].replace({'male': 0, 'female': 1, 'brand': 2})

# Check for unique values in the 'gender' column after replacement
print()
print("Unique Values in 'gender'")
print(df_preprocessed['gender'].unique())
print(df_preprocessed.info())

# Distribution of gender
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df_preprocessed)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('count')
plt.show()

df_gender = df_preprocessed[['gender', 'gender:confidence']].copy()

# Drop the original categorical columns
df_preprocessed = df_preprocessed.drop(columns=categorical_features)

# Convert 'link_color' values
df_preprocessed['link_color_rgb'] = df_preprocessed['link_color'].apply(lambda x: hex_to_rgb(x) if isinstance(x, str) else (0,0,0))
# Convert 'sidebar_color' values
df_preprocessed['sidebar_color_rgb'] = df_preprocessed['sidebar_color'].apply(lambda x: hex_to_rgb(x) if isinstance(x, str) else (0,0,0))

rgb_df = pd.DataFrame(df_preprocessed['link_color_rgb'].to_list(), columns=['link_R', 'link_G', 'link_B'])
rgb_df = pd.concat([rgb_df, pd.DataFrame(df_preprocessed['sidebar_color_rgb'].to_list(), columns=['sidebar_R', 'sidebar_G', 'sidebar_B'])], axis=1)

# Drop the original color features
df_preprocessed = df_preprocessed.drop(columns=['link_color', 'sidebar_color', 'link_color_rgb', 'sidebar_color_rgb'])

# Check if all required features are there
print()
print('All Remaining Features')
print(df_preprocessed.columns.tolist())

# Define the numerical features to scale (filtering for int64 and float64 columns)
numerical_features = df_preprocessed.select_dtypes(include=[np.number])
# print(f'All current numerical features are {numerical_features.columns.tolist()}')

print()
print('Dataset Overview After PreProcessing')
print(df_preprocessed.info())

print()
print('---- NLP Processing ----')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

df_preprocessed['description'].fillna('', inplace=True)
df_preprocessed['text'].fillna('', inplace=True)
# df_preprocessed['name'].fillna('', inplace=True)

# Check the text features if they still contain NaN
print()
print(df_preprocessed.select_dtypes(include=[object]))

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Apply preprocessing to the 'description', 'text', and 'name' columns
df_preprocessed['cleaned_description'] = df_preprocessed['description'].apply(lambda x: preprocess_text(str(x)))
df_preprocessed['cleaned_text'] = df_preprocessed['text'].apply(lambda x: preprocess_text(str(x)))
# df_preprocessed['cleaned_name'] = df_preprocessed['name'].apply(lambda x: preprocess_text(str(x)))

# Check the preprocessed data with preprocessed text features
print(df_preprocessed[['description', 'cleaned_description', 'text', 'cleaned_text']].head())

# Drop the original text features
df_preprocessed = df_preprocessed.drop(columns=['description','text'])

# Initialize TFIDF vectorizer for text features
print()
print('Applying TF-IDF Vectorisation...')
tfidf_vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')

# Apply TF-IDF on 'description', 'text', 'name' columns

tfidf_description = tfidf_vectorizer.fit_transform(df_preprocessed['cleaned_description']).toarray()
tfidf_text = tfidf_vectorizer.fit_transform(df_preprocessed['cleaned_text']).toarray()
# tfidf_name = tfidf_vectorizer.fit_transform(df_preprocessed['cleaned_name']).toarray()

# Convert TF-IDF into DataFrames and add to df_preprocessed
tfidf_desc_df = pd.DataFrame(tfidf_description, columns=[f'desc_{i}' for i in range(tfidf_description.shape[1])])
tfidf_text_df = pd.DataFrame(tfidf_text, columns=[f'text_{i}' for i in range(tfidf_text.shape[1])])
# tfidf_name_df = pd.DataFrame(tfidf_name, columns=[f'name_{i}' for i in range(tfidf_name.shape[1])])

# Merge with main dataframe
df_preprocessed = pd.concat([df_preprocessed.reset_index(drop=True), tfidf_desc_df, tfidf_text_df], axis=1)

# Drop the cleaned text features
df_preprocessed = df_preprocessed.drop(columns=['cleaned_description', 'cleaned_text'])

df_preprocessed = pd.concat([df_preprocessed, rgb_df], axis=1)

df_cate = df_preprocessed[['tweet_location_encoded', 'user_timezone_encoded']].copy()

print()
print('---- CLUSTERING MODELS ----')

print()
print('Exp 1: Using All Selected Features')

sil_ex1 = []
cal_ex1 = []
# Drop the gender and categorical features before normalise

df_cat = df_cate.copy()
# Drop gender feature and categorical features
df_preprocessed = df_preprocessed.drop(columns=df_cat.columns)
df_finalised = df_preprocessed.drop(columns=['gender', 'gender:confidence'])

# Normalise every existing feature
scaler = StandardScaler()
df_finalised = pd.DataFrame(scaler.fit_transform(df_finalised), columns=df_finalised.columns)

df_finalised = pd.concat([df_finalised, df_cat, df_gender], axis=1)
# find the rows that contained NaN values and drop them
df_finalised = df_finalised.dropna()

data_exp1 = df_finalised
df_ex1 = df_finalised.drop(columns=['gender', 'gender:confidence'])


# Check the preprocessed dataset in the present
print()
print('Dataset for Exp 1')
print(df_ex1.info())

# Apply UMAP for dimensionality reduction
umap_model = umap.UMAP()
umap_vis = umap.UMAP(n_neighbors=30,min_dist=0.1, n_components=3, random_state=42)
umap_embedding = umap_model.fit_transform(df_ex1)
umap_plot = umap_vis.fit_transform(df_ex1)
print(umap_embedding.shape)

# K-Means Clustering
print()
print('Performing K-Means Clustering...')
kmeans_clustering = KMeansClustering(umap_embedding)
kmeans_clustering.tune_hyperparameters()
kmeans_exp1 = kmeans_clustering.fit_model()
kmeans_clustering.visualize_clusters(umap_plot, 'All feature types')
kmeans_clustering.plot_elbow_method()
k_labels = kmeans_clustering.output_label()
sil_ex1.append(kmeans_clustering.silhoutte())
cal_ex1.append(kmeans_clustering.calinski())

k_retriever = ClusteringDataRetriever(data_exp1, k_labels)
df_with_labels = k_retriever.get_data_with_labels()

print()
print('Dataset with Labels from KMeans in Exp 1')
print(df_with_labels.head())
for label in np.unique(k_labels):
    print(f'Data points that belong to cluster {label} from KMeans in Exp 1')
    print(k_retriever.get_cluster_data(label))
    print(f'No. of records with gender 0 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 0) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
    print(f'No. of records with gender 1 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 1) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
    print(f'No. of records with gender 2 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 2) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')

# DBSCAN Clustering
print()
print('Performing DBSCAN Clustering...')
dbscan_clustering = DBSCANClustering(umap_embedding)
dbscan_clustering.tune_hyperparameters()
dbscan_exp1 = dbscan_clustering.fit_model()
dbscan_clustering.visualize_clusters_and_outliers_3D(umap_plot, 'All feature types')
db_labels = dbscan_clustering.output_label()
sil_ex1.append(dbscan_clustering.silhoutte())
cal_ex1.append(dbscan_clustering.calinski())

# Initialize the class to retrieve data
db_retriever = ClusteringDataRetriever(data_exp1, db_labels)
df_with_labels = db_retriever.get_data_with_labels()
print()
print('Dataset with Labels from DBSCAN in Exp 1')
print(df_with_labels.head())
for label in np.unique(db_labels):
    if label != -1:
        print(f'Data points that belong to cluster {label} from DBSCAN in Exp 1')
        print(db_retriever.get_cluster_data(label))
        print(f'No. of records with gender 0 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 0) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
        print(f'No. of records with gender 1 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 1) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
        print(f'No. of records with gender 2 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 2) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
print('Data points classified as noise')
print(db_retriever.get_noise_data())

print()
print('Exp 2: Using Only Numerical and Categorical Features')

sil_ex2 = []
cal_ex2 = []

# Normalise every existing feature
scaler = StandardScaler()
chunk_size = 100
for i in range(0, df_num.shape[0], chunk_size):
    df_num.iloc[i:i + chunk_size] = scaler.fit_transform(df_num.iloc[i:i + chunk_size])
df_no_text = pd.concat([df_num, df_cate, df_gender], axis=1)
print("Data with Only Numerical and Categorical Features")
print(df_no_text.head())
print(df_no_text.info())

df_no_text = df_no_text.dropna()
df_no_text_wg = df_no_text.copy()

# Drop gender feature before clustering
data_exp2 = df_no_text.drop(columns=['gender', 'gender:confidence'])

# Check No. of records after drop NaN values
print()
print("Dataset for Exp 2")
print(data_exp2.head())
print(data_exp2.info())

# Apply UMAP for dimensionality reduction
print('Applying UMAP for dim reduction...')
umap_model = umap.UMAP(n_neighbors=30,min_dist=0.1, n_components=3, random_state=42)
umap_embedding = umap_model.fit_transform(data_exp2)
print(umap_embedding.shape)
# umap_embedding = umap_embedding.astype(np.float32)

# K-Means Clustering
print()
print('Performing K-Means Clustering...')
kmeans_clustering = KMeansClustering(data_exp2)
kmeans_clustering.tune_hyperparameters()
kmeans_exp2 = kmeans_clustering.fit_model()
kmeans_clustering.visualize_clusters(umap_embedding, 'Numerical and categorical features')  # Visualize clusters
kmeans_clustering.plot_elbow_method()
k_labels = kmeans_clustering.output_label()
sil_ex2.append(kmeans_clustering.silhoutte())
cal_ex2.append(kmeans_clustering.calinski())

k_retriever = ClusteringDataRetriever(df_no_text_wg, k_labels)
df_with_labels = k_retriever.get_data_with_labels()
print()
print('Dataset with Labels from KMeans in Exp 2')
print(df_with_labels.head())
for label in np.unique(k_labels):
    print(f'Data points that belong to cluster {label} from KMeans in Exp 2')
    print(k_retriever.get_cluster_data(label))
    print(f'No. of records with gender 0 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 0) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
    print(f'No. of records with gender 1 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 1) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
    print(f'No. of records with gender 2 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 2) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')

# DBSCAN Clustering
print()
print('Performing DBSCAN Clustering...')
dbscan_clustering = DBSCANClustering(data_exp2)
dbscan_clustering.tune_hyperparameters()  # Tune DBSCAN hyperparameters
dbscan_exp2 = dbscan_clustering.fit_model()  # Fit the DBSCAN model
dbscan_clustering.visualize_clusters_and_outliers_3D(umap_embedding, 'numerical and categorical features')  # Plot 3D noise points and valid clusters
db_labels = dbscan_clustering.output_label()
sil_ex2.append(dbscan_clustering.silhoutte())
cal_ex2.append(dbscan_clustering.calinski())


db_retriever = ClusteringDataRetriever(df_no_text_wg, db_labels)
df_with_labels = db_retriever.get_data_with_labels()
print()
print('Dataset with Labels from DBSCAN in Exp 2')
print(df_with_labels.head())
for label in np.unique(db_labels):
    if label != -1:
        print(f'Data points that belong to cluster {label} from DBSCAN in Exp 2')
        print(db_retriever.get_cluster_data(label))
        print(f'No. of records with gender 0 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 0) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
        print(f'No. of records with gender 1 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 1) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
        print(f'No. of records with gender 2 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 2) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
print('Data points classified as noise')
print(db_retriever.get_noise_data())

print()
print('Exp 3: Using Only Text Features')

sil_ex3 = []
cal_ex3 = []
# Merge with main dataframe
df_with_text = pd.concat([tfidf_desc_df, tfidf_text_df], axis=1)
# Normalise every existing feature
scaler = StandardScaler()
chunk_size = 100
for i in range(0, df_with_text.shape[0], chunk_size):
    df_with_text.iloc[i:i + chunk_size] = scaler.fit_transform(df_with_text.iloc[i:i + chunk_size])

df_with_text_wg = pd.concat([df_with_text, df_gender], axis=1)
# Drop NaN values before clustering
df_with_text_wg = df_with_text_wg.dropna()
data_exp3 = df_with_text_wg.drop(columns=['gender', 'gender:confidence'])

# Drop the gender features before clustering

print('Dataset for Exp 3')
print(data_exp3.info())
print(data_exp3.head())

umap_model = umap.UMAP()
umap_embedding_t = umap_model.fit_transform(data_exp3)
umap_embedding = umap.UMAP(n_neighbors=30,min_dist=0.1, n_components=3, random_state=42).fit_transform(data_exp3)

# K-Means Clustering
print()
print('Performing K-Means Clustering...')
kmeans_clustering = KMeansClustering(umap_embedding_t)
kmeans_clustering.tune_hyperparameters()
kmeans_exp3 = kmeans_clustering.fit_model()
kmeans_clustering.visualize_clusters(umap_embedding, 'Text features')
kmeans_clustering.plot_elbow_method()
k_labels = kmeans_clustering.output_label()
sil_ex3.append(kmeans_clustering.silhoutte())
cal_ex3.append(kmeans_clustering.calinski())

k_retriever = ClusteringDataRetriever(df_with_text_wg, k_labels)
df_with_labels = k_retriever.get_data_with_labels()
print()
print('Dataset with Labels from KMeans in Exp 3')
print(df_with_labels.head())
for label in np.unique(k_labels):
    print(f'Data points that belong to cluster {label} from KMeans in Exp 3')
    print(k_retriever.get_cluster_data(label))
    print(f'No. of records with gender 0 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 0) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
    print(f'No. of records with gender 1 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 1) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
    print(f'No. of records with gender 2 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 2) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')

# DBSCANClustering
print()
print('Performing DBSCAN Clustering...')
dbscan_clustering = DBSCANClustering(umap_embedding_t)
dbscan_clustering.tune_hyperparameters()
dbscan_exp3 = dbscan_clustering.fit_model()
dbscan_clustering.visualize_clusters_and_outliers_3D(umap_embedding, 'Text features')
db_labels = dbscan_clustering.output_label()
sil_ex3.append(dbscan_clustering.silhoutte())
cal_ex3.append(dbscan_clustering.calinski())

db_retriever = ClusteringDataRetriever(df_with_text_wg, db_labels)
df_with_labels = db_retriever.get_data_with_labels()
print()
print('Dataset with Labels from DBSCAN in Exp 3')
print(df_with_labels.head())
for label in np.unique(db_labels):
    if label != -1:
        print(f'Data points that belong to cluster {label} from DBSCAN in Exp 3')
        print(db_retriever.get_cluster_data(label))
        print(f'No. of records with gender 0 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 0) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
        print(f'No. of records with gender 1 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 1) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
        print(f'No. of records with gender 2 in cluster {label} is {df_with_labels[(df_with_labels["gender"] == 2) & (df_with_labels["Cluster_Label"] == label)].shape[0]}')
print('Data points classified as noise')
print(db_retriever.get_noise_data())

print()
print('---- VISUALIZE THE METRIC EVALUATION ----')

# Metric functions
model_names = ['KMeans', 'DBSCAN']

sil_scores = [sil_ex1, sil_ex2, sil_ex3]
cal_scores = [cal_ex1, cal_ex2, cal_ex3]

plot_silhouette_bar_across_experiments(model_names, sil_scores)
visualize_ch_index_across_experiments(model_names, cal_scores)
