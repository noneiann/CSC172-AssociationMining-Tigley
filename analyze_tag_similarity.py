"""
Analyze tag similarity using sentence transformers
to find potential synonym candidates
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned tags
print("Loading cleaned tags...")
df = pd.read_csv('datasets/cleaned/genome-tags-cleaned.csv')
tags = df['tag'].unique().tolist()
print(f"Found {len(tags)} unique tags")

# Load the model
print("\nLoading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(tags, show_progress_bar=True)

# Calculate similarity matrix
print("Calculating similarity matrix...")
similarity_matrix = cosine_similarity(embeddings)

# Find similar pairs
print("\nFinding similar tag pairs...")
candidates = []
for i in range(len(tags)):
    for j in range(i+1, len(tags)):
        sim = similarity_matrix[i][j]
        if sim > 0.7:  # High similarity threshold
            candidates.append({
                'tag1': tags[i],
                'tag2': tags[j],
                'similarity': sim
            })

# Sort by similarity
candidates_df = pd.DataFrame(candidates).sort_values('similarity', ascending=False)

print(f"\n{'='*80}")
print(f"FOUND {len(candidates_df)} SIMILAR TAG PAIRS (similarity > 0.7)")
print(f"{'='*80}\n")

# Show top 100 candidates
print("TOP 100 MOST SIMILAR TAG PAIRS:\n")
for idx, row in candidates_df.head(100).iterrows():
    print(f"{row['similarity']:.4f} | {row['tag1']:<30} ↔ {row['tag2']}")

# Save full results
output_path = 'datasets/tag_similarity_analysis.csv'
candidates_df.to_csv(output_path, index=False)
print(f"\n✅ Full results saved to: {output_path}")

# Group analysis - find clusters of related tags
print(f"\n{'='*80}")
print("POTENTIAL SYNONYM CLUSTERS")
print(f"{'='*80}\n")

# Find clusters by grouping highly similar tags
processed = set()
clusters = []

for idx, row in candidates_df.iterrows():
    tag1, tag2 = row['tag1'], row['tag2']
    
    # Skip if already in a cluster
    if tag1 in processed and tag2 in processed:
        continue
    
    # Find or create cluster
    found_cluster = None
    for cluster in clusters:
        if tag1 in cluster or tag2 in cluster:
            found_cluster = cluster
            break
    
    if found_cluster:
        found_cluster.add(tag1)
        found_cluster.add(tag2)
    else:
        clusters.append({tag1, tag2})
    
    processed.add(tag1)
    processed.add(tag2)

# Show top clusters
print("Top 30 synonym clusters (tags that should potentially merge):\n")
for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)[:30], 1):
    if len(cluster) > 1:
        print(f"{i}. {', '.join(sorted(cluster))}")

print(f"\n💡 Review these clusters and add to TAG_SYNONYMS in clean_dataset.py")
