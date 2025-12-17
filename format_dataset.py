import pandas as pd

# Load the datasets
print("Loading datasets...")
df_movies = pd.read_csv("datasets/movies.csv")
df_genome_scores = pd.read_csv("datasets/genome-scores-cleaned.csv")
df_genome_tags = pd.read_csv("datasets/genome-tags-cleaned.csv")

print("Processing transactions...")

df_genome = (
    df_genome_scores
    .merge(df_genome_tags, on='tagId')
    .merge(df_movies[['movieId', 'title']], on='movieId', how='left')
)

# Prune tags that appear in too few movies
print(f"Tags before pruning: {df_genome['tag'].nunique()}")
tag_counts = df_genome.groupby('tag')['movieId'].nunique().to_dict()

MIN_MOVIES = 500

valid_tags = {
    tag for tag, count in tag_counts.items()
    if count >= MIN_MOVIES
}

print(f"Tags after pruning (>={MIN_MOVIES} movies): {len(valid_tags)}")

# Filter to keep only valid tags
df_genome = df_genome[df_genome['tag'].isin(valid_tags)]

transactions = []

for movie_id, group in df_genome.groupby('movieId'):
    top_tags = group.nlargest(10, 'relevance')
    
    if len(top_tags) < 5:
        continue
    
    transactions.append({
        'movieId': movie_id,
        'title': top_tags['title'].iloc[0],
        'tags': top_tags['tag'].tolist()
    })

df_transactions = pd.DataFrame(transactions)

print(f"Movies with sufficient valid tags: {len(df_transactions)}")

# Save to CSV
df_transactions.to_csv('datasets/transactions.csv', index=False)

# Display sample transactions
print("\nSample Transactions:")
print(df_transactions.head(10))
print(f"\nTotal transactions: {len(df_transactions)}")
print(f"\nEach transaction contains a movie and its 5 most relevant tags.")

# Also create a more readable format for association mining
print("\nCreating association mining format...")
with open('datasets/transactions_formatted.txt', 'w', encoding='utf-8') as f:
    for _, row in df_transactions.iterrows():
        # Write each transaction as comma-separated tags
        f.write(','.join(row['tags']) + '\n')

print("Saved to: datasets/transactions.csv and datasets/transactions_formatted.txt")