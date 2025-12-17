import pandas as pd

# Load the datasets
print("Loading datasets...")
df_movies = pd.read_csv("datasets/movies.csv")
df_genome_scores = pd.read_csv("datasets/genome-scores.csv")
df_genome_tags = pd.read_csv("datasets/genome-tags.csv")

# Filter relevance scores between 0.2 and 0.8
print(f"Original genome scores count: {len(df_genome_scores)}")
df_genome_scores = df_genome_scores[(df_genome_scores['relevance'] >= 0.2) & (df_genome_scores['relevance'] <= 1)]
print(f"Filtered genome scores count (0.2-0.8): {len(df_genome_scores)}")

# Define stop-tags
STOP_TAGS = {
    'good movie', 'great movie', 'classic', 'favorite', 'excellent',
    'bad movie', 'boring', 'entertaining', 'must see', 'highly recommended',
    'good', 'great', 'bad', 'movie', 'film', 'cinema',
    'watchable', 'seen it', 'own it', 'loved it', 'liked it', 'hated it',
    'overrated', 'underrated', 'interesting', 'recommended', 'original', 'clever'
}

# Merge to get tag names with scores
df_genome = df_genome_scores.merge(df_genome_tags, on='tagId')

print("\n Genome Scores Statistics: ")
print(df_genome_scores['relevance'].describe())

# Analyze tag frequency and average relevance
print("\n Top 30 Most Frequent Tags (by movie count): ")
tag_stats = df_genome.groupby('tag').agg({
    'movieId': 'count',
    'relevance': 'mean'
}).rename(columns={'movieId': 'movie_count', 'relevance': 'avg_relevance'})
tag_stats = tag_stats.sort_values('movie_count', ascending=False)
print(tag_stats.head(30))

# Identify potential stop-tags
print("\nPotential Stop-Tags (Generic/Non-Informative): ")
generic_tags = df_genome[df_genome['tag'].str.contains(
    'good|great|bad|classic|favorite|excellent|boring|must|recommend|overrated|underrated|interesting',
    case=False, na=False
)]['tag'].unique()
print(f"Found {len(generic_tags)} potentially generic tags:")
for tag in sorted(generic_tags)[:50]:
    print(f"  - {tag}")

# Analyze stop-tags if they exist
print("\n Stop-Tag Analysis ")
stop_tags_in_data = [tag for tag in STOP_TAGS if tag in df_genome_tags['tag'].values]
print(f"Stop-tags found in dataset: {len(stop_tags_in_data)}")
if stop_tags_in_data:
    print("Tags:", stop_tags_in_data)
    
# Calculate how many high-relevance scores include generic terms
high_relevance = df_genome[df_genome['relevance'] > 0.5]
generic_high = high_relevance[high_relevance['tag'].str.contains(
    'good|great|bad|classic|favorite|excellent|boring|must|recommend',
    case=False, na=False
)]
print(f"\nHigh-relevance entries (>0.5): {len(high_relevance)}")
print(f"Generic tags in high-relevance: {len(generic_high)} ({len(generic_high)/len(high_relevance)*100:.2f}%)")

print("\n Recommended: Filter out these generic tags")
print("Update STOP_TAGS set based on the analysis above.")
print("Consider filtering tags that appear in nearly all movies or have subjective quality judgments.")