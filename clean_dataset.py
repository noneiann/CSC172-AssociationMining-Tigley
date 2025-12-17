import pandas as pd

# Load the datasets
print("Loading datasets...")
df_movies = pd.read_csv("datasets/movies.csv")
df_genome_scores = pd.read_csv("datasets/genome-scores.csv")
df_genome_tags = pd.read_csv("datasets/genome-tags.csv")

# Filter relevance scores
print(f"Original genome scores count: {len(df_genome_scores)}")
df_genome_scores = df_genome_scores[(df_genome_scores['relevance'] >= 0.4) & (df_genome_scores['relevance'] <= 1)]
print(f"Filtered genome scores count (0.4-1.0): {len(df_genome_scores)}")

# Define stop-tags (from analyze_dataset.py)
STOP_TAGS = {
    'good movie', 'great movie', 'classic', 'favorite', 'excellent',
    'bad movie', 'boring', 'entertaining', 'must see', 'highly recommended',
    'good', 'great', 'bad', 'movie', 'film', 'cinema',
    'watchable', 'seen it', 'own it', 'loved it', 'liked it', 'hated it',
    'overrated', 'underrated', 'interesting', 'recommended', 'original', 'clever'
}

print("Cleaning data...")

# Merge genome scores with tags to enable filtering
df_genome = df_genome_scores.merge(df_genome_tags, on='tagId')

# Identify generic tags dynamically
print(f"Tags before filtering: {df_genome['tag'].nunique()}")
generic_tags = df_genome[df_genome['tag'].str.contains(
    'good|great|bad|classic|favorite|excellent|boring|must|recommend|overrated|underrated|interesting',
    case=False, na=False
)]['tag'].unique()

# Combine manually defined stop-tags with detected generic tags
all_stop_tags = STOP_TAGS.union(set(generic_tags))
print(f"Total stop-tags to filter: {len(all_stop_tags)}")

# Filter out stop-tags
df_genome = df_genome[~df_genome['tag'].isin(all_stop_tags)]
print(f"Tags after filtering stop-tags: {df_genome['tag'].nunique()}")

# Get cleaned genome scores (without the tag names, just the filtered scores)
df_cleaned_scores = df_genome[['movieId', 'tagId', 'relevance']]

# Save cleaned datasets
df_cleaned_scores.to_csv('datasets/genome-scores-cleaned.csv', index=False)
df_genome_tags.to_csv('datasets/genome-tags-cleaned.csv', index=False)

# Display summary
print("\nCleaned Data Summary:")
print(f"Total movie-tag pairs: {len(df_cleaned_scores)}")
print(f"Unique movies: {df_cleaned_scores['movieId'].nunique()}")
print(f"Unique tags: {df_cleaned_scores['tagId'].nunique()}")
print(f"\nRelevance range: {df_cleaned_scores['relevance'].min():.4f} - {df_cleaned_scores['relevance'].max():.4f}")
print(f"Mean relevance: {df_cleaned_scores['relevance'].mean():.4f}")

print("\nSaved cleaned data to:")
print("  - datasets/genome-scores-cleaned.csv")
print("  - datasets/genome-tags-cleaned.csv")
print("\nNext step: Run format_dataset.py to create transactions from cleaned data.")
