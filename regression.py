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
    ('statsmodels', 'statsmodels==0.14.3'),
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

# Load the dataset
df = pd.read_csv('twitter_user_data.csv', encoding='ISO-8859-1')

# Quick view of the dataset
# print('The information of the dataset')
# print(df.info())
# print('The first few rows of the dataset')
# print(df.head())

all_features = df.columns


# Finding features that have a lot of missing data
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


missing_col, df_cleaned = find_columns_with_missing(df, all_features)
missing_col
# print('The information of the cleaned dataset')
# print(df_cleaned.info())
# print('The first few rows of the cleaned dataset')
# print(df_cleaned.head())

# Dropping rows where 'gender' is missing
df_cleaned = df_cleaned.dropna(subset=['gender'])

# Drop the 'profile_yn' column since it is not relevant to human/non-human classification
df_cleaned = df_cleaned.drop(columns=['profile_yn'])

# Now that we have handled the missing data, you can proceed with further analysis
# print('The information of the cleaned dataset')
# print(df_cleaned.info())
# print('The first few rows of the cleaned dataset')
# print(df_cleaned.head())

# Exploratory Data Analysis (EDA)
current_num_features = df.select_dtypes(include=[np.number])

# Plot distribution of each numerical feature with gender as hue using seaborn
# for feature in current_num_features:
    # plt.figure(figsize=(8, 6))
    # sns.histplot(df_cleaned, x=feature, hue='gender', bins=30, kde=True)
    # plt.title(f'Distribution of {feature} by Gender')
    # plt.show()

# Distribution of gender
# plt.figure(figsize=(8, 6))
# sns.countplot(x='gender', data=df_cleaned)
# plt.title('Distribution of Gender')
# plt.xlabel('Gender')
# plt.ylabel('count')
# plt.show()

# Plot distribution of 'tweet_count' and 'retweet_count'
# for column in ['tweet_count', 'retweet_count']:
    # plt.figure(figsize=(8, 6))
    # sns.histplot(data=df_cleaned, x=column, kde=True, bins=30)
    # plt.title(f'Distribution of {column.replace("_", " ").capitalize()}')
    # plt.show()

# Correlation analysis for numerical features
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_cleaned[['tweet_count', 'retweet_count', 'fav_number']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()

# Extracting date from 'created' and 'tweet_created' for time-based analysis
df_cleaned['profile_created_year'] = pd.to_datetime(df_cleaned['created']).dt.year
df_cleaned['tweet_created_year'] = pd.to_datetime(df_cleaned['tweet_created']).dt.year

# Ensure 'created' and tweet_created are in datetime format
df_cleaned['created'] = pd.to_datetime(df_cleaned['created'], errors='coerce')
df_cleaned['tweet_created'] = pd.to_datetime(df_cleaned['tweet_created'], errors='coerce')

#assuming the data was up-to-date
df_cleaned['account_age'] = (pd.Timestamp.now() - df_cleaned['created']).dt.days

df_cleaned['tweets_per_day'] = df_cleaned['tweet_count'] / df_cleaned['account_age']
df_cleaned['retweets_per_day'] = df_cleaned['retweet_count'] / df_cleaned['account_age']
df_cleaned['favorites_per_day'] = df_cleaned['fav_number'] / df_cleaned['account_age']

# Plotting the distribution of profile creation over the years
# plt.figure(figsize=(8, 6))
# sns.histplot(df_cleaned['profile_created_year'], kde=False, bins=15)
# plt.title('Distribution of Profile Creation Years')
# plt.xlabel('Profile Created Year')
# plt.ylabel('count')
# plt.show()

# Plotting the histogram of tweets per day
# plt.figure(figsize=(10, 6))
# sns.histplot(df_cleaned['tweets_per_day'], bins=50, kde=True)
# plt.title('Distribution of Tweets Per Day')
# plt.xlabel('Tweets Per Day')
# plt.ylabel('Frequency')
# plt.show()

#show the relationship between account age and tweets per day
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='account_age', y='tweets_per_day', data=df_cleaned)
# plt.title('Account Age vs. Tweets Per Day')
# plt.xlabel('Account Age (Days)')
# plt.ylabel('Tweets Per Day')
# plt.show()

# Exploring 'link_color' and 'sidebar_color' features

#Check number of NaN value in  'link_color' and 'sidebar_color' features
link_color_nan_count = df_cleaned['link_color'].isnull().sum()
sidebar_color_nan_count = df_cleaned['sidebar_color'].isnull().sum()

# print(f"Number of NaN values in 'link_color': {link_color_nan_count}")
# print(f"Number of NaN values in 'sidebar_color': {sidebar_color_nan_count}")

#Check how many available colors in 'link_color' and 'sidebar_color' features
link_color_count = len(df_cleaned['link_color'].unique())
sidebar_color_count = len(df_cleaned['sidebar_color'].unique())
# print(f'the number of link color is {link_color_count}')
# print(f'the number of side bar color is {sidebar_color_count}')

# Apply the function to 'link_color' and 'sidebar_color'
df_cleaned['link_color'] = df_cleaned['link_color'].apply(lambda x: f'#{x}' if len(x) == 6 else '#000000')
df_cleaned['sidebar_color'] = df_cleaned['sidebar_color'].apply(lambda x: f'#{x}' if len(x) == 6 else '#000000')

# Drop rows where 'sidebar_color' is still NaN
df_cleaned = df_cleaned.dropna(subset=['link_color'])
df_cleaned = df_cleaned.dropna(subset=['sidebar_color'])
# print(f"Number of NaN values in 'link_color': {df_cleaned['link_color'].isnull().sum()}")
# print(f"Number of NaN values in 'sidebar_color': {df_cleaned['sidebar_color'].isnull().sum()}")

#top 15 colors
top_sidebar_colors = df_cleaned['sidebar_color'].value_counts().iloc[:15].index.tolist()
top_link_colors = df_cleaned['link_color'].value_counts().iloc[:15].index.tolist()
#print(top_sidebar_colors)

# Extract top 10 most common sidebar colors 
# sns.set(rc={'axes.facecolor':'lightgrey', 'figure.facecolor':'white'})
# plt.figure(figsize=(8, 6))
# sns.countplot(y='sidebar_color', data=df_cleaned, order=df_cleaned['sidebar_color'].value_counts().iloc[:15].index, palette=top_sidebar_colors)
# plt.title('Top 15 Most Common Profile sidebar_color')
# plt.ylabel('Sidebar Color')
# plt.xlabel('count')
# plt.grid()
# plt.show()

# Extract top 10 most common link colors 
# sns.set(rc={'axes.facecolor':'lightgrey', 'figure.facecolor':'white'})
# plt.figure(figsize=(8, 6))
# sns.countplot(y='link_color', data=df_cleaned, order=df_cleaned['link_color'].value_counts().iloc[:15].index, palette=top_link_colors)
# plt.title('Top 15 Most Common Profile link_color')
# plt.ylabel('Link Color')
# plt.xlabel('count')
# plt.grid()
# plt.show()

# count plot for sidebar_color vs. gender
# plt.figure(figsize=(10, 6))
# sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
# sns.countplot(x='sidebar_color', hue='gender', data=df_cleaned, order=df_cleaned['sidebar_color'].value_counts().iloc[:15].index)
# plt.title('Top 15 Most Common Sidebar Colors by Gender')
# plt.xlabel('Sidebar Color')
# plt.ylabel('count')
# plt.xticks(rotation=45)
# plt.show()

# count plot for link_color vs. gender
# plt.figure(figsize=(10, 6))
# sns.countplot(x='link_color', hue='gender', data=df_cleaned, order=df_cleaned['link_color'].value_counts().iloc[:15].index)
# plt.title('Top 15 Most Common link Colors by Gender')
# plt.xlabel('Link Color')
# plt.ylabel('count')
# plt.xticks(rotation=45)
# plt.show()

# Scatter plot for link_color vs. tweet_count with gender as hue
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='link_color', y='tweet_count', hue='gender', data=df_cleaned[df_cleaned['link_color'].isin(top_link_colors)], palette='Set2', s=100, alpha=0.7)
# plt.title('Link Colors vs. Tweet count with Gender')
# plt.xlabel('Link Color')
# plt.ylabel('Tweet count')
# plt.xticks(rotation=45)
# plt.show()

# Scatter plot for sidebar_color vs. tweet_count with gender as hue
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='sidebar_color', y='tweet_count', hue='gender', data=df_cleaned[df_cleaned['sidebar_color'].isin(top_sidebar_colors)], palette='Set2', s=100, alpha=0.7)
# plt.title('Sidebar Colors vs. Tweet count with Gender')
# plt.xlabel('Sidebar Color')
# plt.ylabel('Tweet count')
# plt.xticks(rotation=45)
# plt.show()

# Select columns to be used
col = ['gender', 'gender:confidence', 'description', 'favorites_per_day','link_color',
       'retweets_per_day', 'sidebar_color', 'text', 'tweets_per_day','user_timezone', 'tweet_location', 'profile_created_year', 'tweet_created_year'
       ]
df_preprocessed = df_cleaned[col].copy()
# Remove rows where gender is 'Unknown'
df_preprocessed = df_preprocessed[df_preprocessed['gender'] != 'unknown']

# Plot correlation matrix
corr_matrix = df_preprocessed.select_dtypes(include=[np.number]).corr()
# sns.heatmap(corr_matrix, annot=True)
# plt.show()

# Drop one feature from highly correlated pairs (correlation > 0.9)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
df_preprocessed = df_preprocessed.drop(columns=to_drop)

# Filling missing values for important features
df_preprocessed['user_timezone'].fillna('Unknown', inplace=True)
df_preprocessed['tweet_location'].fillna('Unknown', inplace=True)
categorical_features = ['user_timezone', 'tweet_location']

#categorise types of features

#numerical features
df_num = df_preprocessed[['retweets_per_day', 'favorites_per_day', 'tweets_per_day', 'profile_created_year', 'tweet_created_year']].copy()

#categorical features with frequency encoding
freq_encoding_location = df_preprocessed['tweet_location'].value_counts(normalize=True)
df_preprocessed['tweet_location_encoded'] = df_preprocessed['tweet_location'].map(freq_encoding_location)

freq_encoding_timezone = df_preprocessed['user_timezone'].value_counts(normalize=True)
df_preprocessed['user_timezone_encoded'] = df_preprocessed['user_timezone'].map(freq_encoding_timezone)

df_cate = df_preprocessed[['tweet_location_encoded', 'user_timezone_encoded']].copy()

#gender features
#encode the 'gender' column to numeric values
df_preprocessed['gender'] = df_preprocessed['gender'].replace({'male': 0, 'female': 1, 'brand': 2})

# Check for unique values in the 'gender' column after replacement
# print(df_preprocessed['gender'].unique())
# print(df_preprocessed.info())

# Distribution of gender
# plt.figure(figsize=(8, 6))
# sns.countplot(x='gender', data=df_preprocessed)
# plt.title('Distribution of Gender')
# plt.xlabel('Gender')
# plt.ylabel('count')
# plt.show()

df_gender = df_preprocessed[['gender', 'gender:confidence']].copy()

# Drop the original categorical columns
df_preprocessed = df_preprocessed.drop(columns=categorical_features)

# Function to convert hex to RGB
def hex_to_rgb(hex_color):
    # Remove the '#' if it exists
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to integer and split into RGB components
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

# Convert 'link_color' values
df_preprocessed['link_color_rgb'] = df_preprocessed['link_color'].apply(lambda x: hex_to_rgb(x) if isinstance(x, str) else (0,0,0))
# Convert 'sidebar_color' values
df_preprocessed['sidebar_color_rgb'] = df_preprocessed['sidebar_color'].apply(lambda x: hex_to_rgb(x) if isinstance(x, str) else (0,0,0))

rgb_df = pd.DataFrame(df_preprocessed['link_color_rgb'].to_list(), columns=['link_R', 'link_G', 'link_B'])
rgb_df = pd.concat([rgb_df, pd.DataFrame(df_preprocessed['sidebar_color_rgb'].to_list(), columns=['sidebar_R', 'sidebar_G', 'sidebar_B'])], axis=1)

#Drop the original color features
df_preprocessed = df_preprocessed.drop(columns=['link_color', 'sidebar_color', 'link_color_rgb', 'sidebar_color_rgb'])

#keep the gender confidence preprocessed to be able to use it in regression task
preprocessed_gender_conf  = df_preprocessed["gender:confidence"].copy()

#Check if all required features are there
# print(f'All features that will be used are {df_preprocessed.columns.tolist()}')

# Define the numerical features to scale (filtering for int64 and float64 columns)
numerical_features = df_preprocessed.select_dtypes(include=[np.number])
#print(f'All current numerical features are {numerical_features.columns.tolist()}')

# print('After all, here is the information of the dataset')
# print(df_preprocessed.info())

# NLP Processing
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

df_preprocessed['description'].fillna('', inplace=True)
df_preprocessed['text'].fillna('', inplace=True)
#df_preprocessed['name'].fillna('', inplace=True)

#Check the text features if they still contain NaN
# print(df_preprocessed.select_dtypes(include=[object]))


# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    #Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removes punctuation
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    #Tokenize the text
    tokens = word_tokenize(text)
    #Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    #Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    #Join tokens back into a string
    return ' '.join(tokens)

# Apply preprocessing to the 'description', 'text', and 'name' columns
df_preprocessed['cleaned_description'] = df_preprocessed['description'].apply(lambda x: preprocess_text(str(x)))
df_preprocessed['cleaned_text'] = df_preprocessed['text'].apply(lambda x: preprocess_text(str(x)))
#df_preprocessed['cleaned_name'] = df_preprocessed['name'].apply(lambda x: preprocess_text(str(x)))

# Check the preprocessed data with preprocessed text features
# print(df_preprocessed[['description', 'cleaned_description', 'text', 'cleaned_text']].head())

#Drop the original text features
df_preprocessed = df_preprocessed.drop(columns=['description','text'])

#Check the preprocessed dataset in the present
# print('The current information of pre-processed dataset before text preprocessing')
# print(df_preprocessed.info())


# Initialize TFIDF vectorizer for text features
tfidf_vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')

# Apply TF-IDF on 'description', 'text', 'name' columns

tfidf_description = tfidf_vectorizer.fit_transform(df_preprocessed['cleaned_description']).toarray()
tfidf_text = tfidf_vectorizer.fit_transform(df_preprocessed['cleaned_text']).toarray()
#tfidf_name = tfidf_vectorizer.fit_transform(df_preprocessed['cleaned_name']).toarray()

# Convert TF-IDF into DataFrames and add to df_preprocessed
tfidf_desc_df = pd.DataFrame(tfidf_description, columns=[f'desc_{i}' for i in range(tfidf_description.shape[1])])
tfidf_text_df = pd.DataFrame(tfidf_text, columns=[f'text_{i}' for i in range(tfidf_text.shape[1])])
#tfidf_name_df = pd.DataFrame(tfidf_name, columns=[f'name_{i}' for i in range(tfidf_name.shape[1])])

# Merge with main dataframe
df_preprocessed = pd.concat([df_preprocessed.reset_index(drop=True), tfidf_desc_df, tfidf_text_df], axis=1)

#Drop the cleaned text features
df_preprocessed = df_preprocessed.drop(columns=['cleaned_description', 'cleaned_text'])

df_preprocessed = pd.concat([df_preprocessed, rgb_df], axis=1)

# print(df_preprocessed.head())

#================================================REGRSSION CODE============================
## Complete py code ¨

#finish preprocessing for regression
df_preprocessed_reg = df_preprocessed.copy()
y = df_preprocessed["gender:confidence"].reset_index(drop=True)
df_preprocessed_reg = df_preprocessed_reg.drop(['gender', "gender:confidence"], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed_reg, y, test_size=0.6, random_state=42)
boosted_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model
boosted_reg.fit(X_train, y_train)

# Make predictions
y_pred = boosted_reg.predict(X_test)
y_pred_train = boosted_reg.predict(X_train)
y_tot_pred = boosted_reg.predict(df_preprocessed_reg)

# Evaluate performance using Mean Squared Error
mse_test = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_total = mean_squared_error(y, y_tot_pred)

print(f"Mean Squared Error (Train): {mse_train:.4f}")
print(f"Mean Squared Error (Test): {mse_test:.4f}")
print(f"Mean Squared Error (Total): {mse_total:.4f}")

# PLOT MSE
labels = ['Train', 'Test', 'Total']
mse_values = [mse_train, mse_test, mse_total]
plt.figure(figsize=(8, 6))
plt.bar(labels, mse_values, color=['skyblue', 'salmon', 'lightgreen'])
plt.suptitle('Boosted Regression Tree with Vectorised Text/Desc Features', fontsize=16)
plt.title('Mean Squared Error Comparison', fontsize=14)
plt.xlabel('Dataset Type')
plt.ylabel('MSE')
plt.show()

#FEATURE IMPORTANCE
print()
print("=" * 50)
print("Feature Importance Analysis")
print("=" * 50)
# Find column indices that start with 'desc_' and 'text_'
desc_columns = [i for i, col in enumerate(df_preprocessed_reg.columns) if col.startswith('desc_')]
text_columns = [i for i, col in enumerate(df_preprocessed_reg.columns) if col.startswith('text_')]
# Access the corresponding elements from the ndarray using the column indices
desc_array = boosted_reg.feature_importances_[desc_columns]
text_array = boosted_reg.feature_importances_[text_columns]
# Output the results
print("desc_ column indices:", desc_columns)
print("text_ column indices:", text_columns)
print("desc_ array:\n", desc_array)
print("text_ array:\n", text_array)
# Sum the values for desc_ and text_ columns
desc_sum = np.sum(boosted_reg.feature_importances_[desc_columns])
text_sum = np.sum(boosted_reg.feature_importances_[text_columns])
# Create a new DataFrame
new_data = {}
# Add the 'desc' and 'text' columns with the summed values
new_data['desc'] = [desc_sum]
new_data['text'] = [text_sum]
boosted_reg.feature_importances_
# Add the other feature columns that are not desc_ or text_
other_columns = [i for i in range(len(df_preprocessed_reg.columns)) if i not in desc_columns and i not in text_columns]
for i in other_columns:
    col_name = df_preprocessed_reg.columns[i]
    new_data[col_name] = [boosted_reg.feature_importances_[i]]
# Convert the new_data dictionary to a DataFrame
feature_importance = pd.DataFrame(new_data)
# Output the results
print(feature_importance)

#Plot feature importance
df_melted = feature_importance.melt(var_name='Feature', value_name='Importance in percentage')
df_melted = df_melted.sort_values(ascending=False, by="Importance in percentage")
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance in percentage', y='Feature', data=df_melted, palette='viridis')
plt.suptitle('Boosted Regression Tree with Vectorised Text/Desc Features', fontsize=16)
plt.title('Feature Importance Analysis', fontsize=14)
plt.show()


#preprocess dataset for plots with regression results
df_preprocessed_diff = df_preprocessed_reg.copy()
df_preprocessed_diff['difference'] = (y.to_numpy() - y_tot_pred)
df_preprocessed_diff["gender_confidence_pred"] = y_tot_pred
y_reset = y.reset_index(drop=True)
df_preprocessed_diff["gender:confidence"] = y_reset

#filtering out coloumns that might be false mistaken
misclassified_df_reg = df_preprocessed_diff[(df_preprocessed_diff["difference"] > 0.1) & (df_preprocessed_diff["gender_confidence_pred"] < 0.85)]
misclassified_df = df_preprocessed_diff[(df_preprocessed_diff["difference"] > 0.1) & (df_preprocessed_diff["gender_confidence_pred"] < 0.85)]
non_train_misclassify = misclassified_df[misclassified_df.index.isin(X_train.index)]
train_misclassify = misclassified_df[~misclassified_df.index.isin(X_train.index)]

#plotting these columns

def scatterplot_mistaken_points(misclassified_df, X_train):
    # Edit misclassified_df to include 'in X_train'
    misclassified_df["in X_train"] = misclassified_df.index.isin(X_train.index)
    # Create subsets for the two plots
    df_in_X_train = misclassified_df[misclassified_df["in X_train"]]
    df_not_in_X_train = misclassified_df[~misclassified_df["in X_train"]]
    # Set up the matplotlib figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Plot 1: Points in X_train
    sns.scatterplot(data=df_in_X_train, x='gender:confidence', y='gender_confidence_pred', alpha=0.4, ax=axes[0], color='blue')
    axes[0].plot([df_in_X_train['gender:confidence'].min(), df_in_X_train['gender:confidence'].max()],
                [df_in_X_train['gender:confidence'].min(), df_in_X_train['gender:confidence'].max()], 'k--', lw=2)
    axes[0].set_xlabel('Dataset Gender Confidence')
    axes[0].set_ylabel('Predicted Gender Confidence')
    axes[0].set_title(f'In X_train\nTotal Samples: {len(df_in_X_train)}')
    # Plot 2: Points not in X_train
    sns.scatterplot(data=df_not_in_X_train, x='gender:confidence', y='gender_confidence_pred', alpha=0.4, ax=axes[1], color='red')
    axes[1].plot([df_not_in_X_train['gender:confidence'].min(), df_not_in_X_train['gender:confidence'].max()],
                [df_not_in_X_train['gender:confidence'].min(), df_not_in_X_train['gender:confidence'].max()], 'k--', lw=2)
    axes[1].set_xlabel('Dataset Gender Confidence')
    axes[1].set_ylabel('Predicted Gender Confidence')
    axes[1].set_title(f'Not in X_train\nTotal Samples: {len(df_not_in_X_train)}')
    plt.tight_layout()
    plt.show()

def scatter_plot(y, y_tot_pred, model):
    #Plotting more results results
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_tot_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.suptitle(model, fontsize=16)
    plt.title('Gender Confidence Comparison', fontsize=14)
    plt.show()

scatterplot_mistaken_points(misclassified_df, X_train)
scatter_plot(y, y_tot_pred, "Boosted Regression Tree with Vectorised Text/Desc Features")

#==============================analyze without text features=============================================
columns_to_drop = [col for col in df_preprocessed_reg.columns if col.startswith(('desc_', 'text_'))]
df_preprocessed_non_text = df_preprocessed_reg.drop(columns=columns_to_drop)
print(df_preprocessed_non_text)

boosted_reg_non_text = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
# Split the dataset into training and testing sets
X_train_non_text, X_test_non_text, y_train_non_text, y_test_non_text = train_test_split(df_preprocessed_non_text, y, test_size=0.6, random_state=42)
# Fit the model
boosted_reg_non_text.fit(X_train_non_text, y_train_non_text)
# Make predictions
y_pred = boosted_reg_non_text.predict(X_test_non_text)
y_pred_train = boosted_reg_non_text.predict(X_train_non_text)

# Evaluate performance using Mean Squared Error
mse_test = mean_squared_error(y_test_non_text, y_pred)
mse_train = mean_squared_error(y_train_non_text, y_pred_train)
mse_total = mean_squared_error(y, y_tot_pred)
y_tot_pred = boosted_reg_non_text.predict(df_preprocessed_non_text)

print(f"Mean Squared Error (Train): {mse_train:.4f}")
print(f"Mean Squared Error (Test): {mse_test:.4f}")
print(f"Mean Squared Error (Total): {mse_total:.4f}")

# PLOT MSE
labels = ['Train', 'Test', 'Total']
mse_values = [mse_train, mse_test, mse_total]
plt.figure(figsize=(8, 6))
plt.bar(labels, mse_values, color=['skyblue', 'salmon', 'lightgreen'])
plt.suptitle('Boosted Regression Tree without Vectorised Text/Desc Features', fontsize=16)
plt.title('Mean Squared Error Comparison', fontsize=14)
plt.xlabel('Dataset Type')
plt.ylabel('MSE')
plt.show()

# Get feature importances and plot from the model
feature_importances = boosted_reg_non_text.feature_importances_
column_names = X_train_non_text.columns
feature_importance_df = pd.DataFrame({
    'Feature': column_names,
    'Importance in percentage': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance in percentage', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance in percentage', y='Feature', data=feature_importance_df, palette='viridis')
plt.suptitle('Boosted Regression Tree without Vectorised Text/Desc Features', fontsize=16)
plt.title('Feature Importance Analysis', fontsize=14)
plt.show()

#adding the dataset gender confidence
df_preprocessed_non_text["gender_confidence_pred"] = y_tot_pred
y_reset = y.reset_index(drop=True)
df_preprocessed_non_text["gender:confidence"] = y_reset

#Inspecting coulumns that could be suspicous
df_preprocessed_non_text["difference"] = y.to_numpy() - y_tot_pred
misclassified_df = df_preprocessed_non_text[(df_preprocessed_non_text["difference"] > 0.1) & (df_preprocessed_non_text["gender_confidence_pred"] < 0.85)]
non_train_misclassify = misclassified_df[misclassified_df.index.isin(X_train_non_text.index)]
train_misclassify = misclassified_df[~misclassified_df.index.isin(X_train_non_text.index)]
scatterplot_mistaken_points(misclassified_df, X_train_non_text)
scatter_plot(y, y_tot_pred, "Boosted Regression Tree without Vectorised Text/Desc Features")

#====================================Analyzing with a linear regression (Least Squares Implementation)====================
X_train_lin = sm.add_constant(X_train)
X_test_lin = sm.add_constant(X_test)
df_preprocessed_lin = sm.add_constant(df_preprocessed_reg)
model = sm.OLS(y_train, X_train_lin)  # Ordinary least squares (unregularized)
results = model.fit()

#run predictions
y_lin_pred = results.predict(X_test_lin)
y_lin_tot_pred = results.predict(df_preprocessed_lin)
y_lin_train = results.predict(X_train_lin)

# Evaluate performance using Mean Squared Error
mse_test = mean_squared_error(y_test, y_lin_pred)
mse_total = mean_squared_error(y, y_lin_tot_pred)
mse_train = mean_squared_error(y_train, y_lin_train)

print(f"Mean Squared Error (Train): {mse_train:.4f}")
print(f"Mean Squared Error (Test): {mse_test:.4f}")
print(f"Mean Squared Error (Total): {mse_total:.4f}")

# PLOT MSE
labels = ['Train', 'Test', 'Total']
mse_values = [mse_train, mse_test, mse_total]
plt.figure(figsize=(8, 6))
plt.bar(labels, mse_values, color=['skyblue', 'salmon', 'lightgreen'])
plt.suptitle('Linear Regression Tree with Vectorised Textual Features', fontsize=16)
plt.title('Mean Squared Error Comparison', fontsize=14)
plt.xlabel('Dataset Type')
plt.ylabel('MSE')
plt.show()

#final preprocess
df_preprocessed_lin["difference"] = y.to_numpy() - y_lin_tot_pred
y_reset = y.reset_index(drop=True)
df_preprocessed_lin["gender:confidence"] = y
df_preprocessed_lin["gender_confidence_pred"] = y_lin_tot_pred


#identify mistaken users
misclassified_df = df_preprocessed_lin[(df_preprocessed_lin["difference"] > 0.1) & (df_preprocessed_lin["gender_confidence_pred"] < 0.85)]
non_train_misclassify = misclassified_df[misclassified_df.index.isin(X_train_lin.index)]
train_misclassify = misclassified_df[~misclassified_df.index.isin(X_train_lin.index)]

scatter_plot(y, y_lin_tot_pred, "Linear Regression with Vectorised Text/Desc Features")

# Edit misclassified_df to include 'in X_train'
misclassified_df["in X_train"] = misclassified_df.index.isin(X_train_lin.index)
# Create subsets for the two plots
df_in_X_train = misclassified_df[misclassified_df["in X_train"]]
df_not_in_X_train = misclassified_df[~misclassified_df["in X_train"]]

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Plot 1: Points in X_train
sns.scatterplot(data=df_in_X_train, x='gender:confidence', y='gender_confidence_pred', alpha=0.4, ax=axes[0], color='blue')
axes[0].plot([df_in_X_train['gender:confidence'].min(), df_in_X_train['gender:confidence'].max()],
             [df_in_X_train['gender:confidence'].min(), df_in_X_train['gender:confidence'].max()], 'k--', lw=2)
axes[0].set_xlabel('Dataset Gender Confidence')
axes[0].set_ylabel('Predicted Gender Confidence')
axes[0].set_title(f'In X_train\nTotal Samples: {len(df_in_X_train)}')

# Plot 2: Points not in X_train
sns.scatterplot(data=df_not_in_X_train, x='gender:confidence', y='gender_confidence_pred', alpha=0.4, ax=axes[1], color='red')
axes[1].plot([df_not_in_X_train['gender:confidence'].min(), df_not_in_X_train['gender:confidence'].max()],
             [df_not_in_X_train['gender:confidence'].min(), df_not_in_X_train['gender:confidence'].max()], 'k--', lw=2)
axes[1].set_xlabel('Dataset Gender Confidence')
axes[1].set_ylabel('Predicted Gender Confidence')
axes[1].set_title(f'Not in X_train\nTotal Samples: {len(df_not_in_X_train)}')

# Adjust layout
plt.tight_layout()


#================================Identity final mistaken samples====================================
common_samples = misclassified_df_reg.index.intersection(misclassified_df.index)
common_df = misclassified_df.loc[common_samples]

scatterplot_mistaken_points(common_df, X_train_lin)
