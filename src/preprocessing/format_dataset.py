import pandas as pd

# Load the datasets
print("Loading datasets...")
df_movies = pd.read_csv("datasets/raw/movies.csv")
df_genome_scores = pd.read_csv("datasets/cleaned/genome-scores-cleaned.csv")
df_genome_tags = pd.read_csv("datasets/cleaned/genome-tags-cleaned.csv")

print("Processing transactions...")

df_genome = (
    df_genome_scores
    .merge(df_genome_tags, on='tagId')
    .merge(df_movies[['movieId', 'title']], on='movieId', how='left')
)

# Configuration
MIN_MOVIES = 500
TOP_N_TAGS = 10   
MIN_TAGS = 3      

# Prune tags that appear in too few movies
print(f"Tags before pruning: {df_genome['tag'].nunique()}")
tag_counts = df_genome.groupby('tag')['movieId'].nunique()

print(f"\nTag frequency distribution:")
print(f"  Tags in 1000+ movies: {(tag_counts >= 1000).sum()}")
print(f"  Tags in 500-999 movies: {((tag_counts >= 500) & (tag_counts < 1000)).sum()}")
print(f"  Tags in 100-499 movies: {((tag_counts >= 100) & (tag_counts < 500)).sum()}")
print(f"  Tags in <100 movies: {(tag_counts < 100).sum()}")

valid_tags = {
    tag for tag, count in tag_counts.items()
    if count >= MIN_MOVIES
}

print(f"\nTags after pruning (>={MIN_MOVIES} movies): {len(valid_tags)}")

# Filter to keep only valid tags
df_genome = df_genome[df_genome['tag'].isin(valid_tags)]

# Calculate tag IDF (inverse document frequency) for diversity
import numpy as np
total_movies = df_genome['movieId'].nunique()
tag_idf = {tag: np.log(total_movies / count) for tag, count in tag_counts.items()}

transactions = []

for movie_id, group in df_genome.groupby('movieId'):
    # Add diversity score: relevance * IDF (prefer unique tags)
    group = group.copy()
    group['diversity_score'] = group['relevance'] * group['tag'].map(tag_idf).fillna(0)
    
    # Select top tags by relevance, but also ensure diversity
    top_tags = group.nlargest(TOP_N_TAGS, 'relevance')
    
    if len(top_tags) < MIN_TAGS:
        continue
    
    transactions.append({
        'movieId': movie_id,
        'title': top_tags['title'].iloc[0],
        'tags': top_tags['tag'].tolist()
    })

df_transactions = pd.DataFrame(transactions)

print(f"Movies with sufficient valid tags: {len(df_transactions)}")

# Save to CSV
df_transactions.to_csv('datasets/formatted/transactions.csv', index=False)

# Display sample transactions
print("\nSample Transactions:")
print(df_transactions.head(10))
print(f"\nTotal transactions: {len(df_transactions)}")

# Analyze tag distribution
all_tags = [tag for tags in df_transactions['tags'] for tag in tags]
from collections import Counter
tag_freq = Counter(all_tags)
print(f"\nTransaction Statistics:")
print(f"  Avg tags per movie: {len(all_tags) / len(df_transactions):.1f}")
print(f"  Unique tags used: {len(tag_freq)}")
print(f"  Most common tags: {', '.join([f'{tag}({count})' for tag, count in tag_freq.most_common(5)])}")