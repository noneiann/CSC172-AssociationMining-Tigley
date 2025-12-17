import pandas as pd

# Load the datasets
print("Loading datasets...")
df_movies = pd.read_csv("datasets/movies.csv")
df_genome_scores = pd.read_csv("datasets/genome-scores.csv")
df_genome_tags = pd.read_csv("datasets/genome-tags.csv")

print("Processing transactions...")
df_genome = df_genome_scores.merge(df_genome_tags, on='tagId')

# For each movie, get the top 5 most relevant tags
transactions = []
for movie_id in df_genome['movieId'].unique():
    # Get all tags for this movie
    movie_tags = df_genome[df_genome['movieId'] == movie_id]
    
    # Sort by relevance and get top 5
    top_tags = movie_tags.nlargest(10, 'relevance')
    
    # Create a transaction (list of tags for this movie)
    tag_list = top_tags['tag'].tolist()
    
    # Get movie title
    movie_title = df_movies[df_movies['movieId'] == movie_id]['title'].values
    movie_title = movie_title[0] if len(movie_title) > 0 else f"Movie {movie_id}"
    
    transactions.append({
        'movieId': movie_id,
        'title': movie_title,
        'tags': tag_list
    })

# Create DataFrame
df_transactions = pd.DataFrame(transactions)

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