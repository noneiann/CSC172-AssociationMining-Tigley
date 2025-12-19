"""
Visualization Tools for Association Mining Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx
from itertools import combinations

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_rule_scatter(rules_df, figsize=(12, 8)):
    """
    Scatter plot of association rules: Confidence vs Lift
    Color by support, size by antecedent length
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate antecedent size for point sizing
    rules_df['ant_size'] = rules_df['antecedents'].apply(lambda x: len(x))
    
    scatter = ax.scatter(
        rules_df['confidence'], 
        rules_df['lift'],
        c=rules_df['support'],
        s=rules_df['ant_size'] * 50,
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Support', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lift', fontsize=14, fontweight='bold')
    ax.set_title('Association Rules: Confidence vs Lift\n(Size=Antecedent Length, Color=Support)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add reference lines
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Lift=1 (No association)')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Conf=0.7 (High confidence)')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_top_rules_network(rules_df, top_n=20, figsize=(16, 12)):
    """
    Network graph showing top association rules
    """
    top_rules = rules_df.nlargest(top_n, 'lift')
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges (rules)
    for _, row in top_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for ant in antecedents:
            for con in consequents:
                # Edge weight = lift, thickness = confidence
                G.add_edge(ant, con, 
                          weight=row['lift'], 
                          confidence=row['confidence'],
                          support=row['support'])
    
    # Calculate node sizes based on degree
    node_sizes = [G.degree(node) * 300 + 500 for node in G.nodes()]
    
    # Get edge weights for coloring
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    confidences = [G[u][v]['confidence'] for u, v in edges]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          edgecolors='black',
                          linewidths=2,
                          alpha=0.9,
                          ax=ax)
    
    # Draw edges with varying thickness
    edge_widths = [conf * 5 for conf in confidences]
    
    edges_drawn = nx.draw_networkx_edges(G, pos,
                                         width=edge_widths,
                                         alpha=0.6,
                                         edge_color=weights,
                                         edge_cmap=plt.cm.RdYlGn,
                                         arrows=True,
                                         arrowsize=20,
                                         arrowstyle='->',
                                         connectionstyle='arc3,rad=0.1',
                                         ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold',
                           font_family='sans-serif',
                           ax=ax)
    
    # Add colorbar for edge colors (lift)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                               norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Lift', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_title(f'Top {top_n} Association Rules Network\n(Node size=Degree, Edge thickness=Confidence, Edge color=Lift)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_tag_cooccurrence_heatmap(transactions_df, top_n=30, figsize=(14, 12)):
    """
    Heatmap showing tag co-occurrence frequencies
    """
    # Get all tags
    all_tags = []
    for tags in transactions_df['tags']:
        all_tags.extend(tags)
    
    # Get most common tags
    tag_counter = Counter(all_tags)
    top_tags = [tag for tag, _ in tag_counter.most_common(top_n)]
    
    # Create co-occurrence matrix
    cooccurrence = pd.DataFrame(0, index=top_tags, columns=top_tags, dtype=int)
    
    for tags in transactions_df['tags']:
        tag_set = [tag for tag in tags if tag in top_tags]
        for tag1, tag2 in combinations(tag_set, 2):
            cooccurrence.loc[tag1, tag2] += 1
            cooccurrence.loc[tag2, tag1] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cooccurrence, 
                annot=False,
                cmap='YlOrRd',
                cbar_kws={'label': 'Co-occurrence Count'},
                square=True,
                linewidths=0.5,
                linecolor='white',
                ax=ax)
    
    ax.set_title(f'Tag Co-occurrence Heatmap (Top {top_n} Tags)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tags', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tags', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    return fig


def plot_tag_frequency_distribution(transactions_df, figsize=(14, 6)):
    """
    Bar chart and histogram of tag frequency distribution
    """
    # Count all tags
    all_tags = []
    for tags in transactions_df['tags']:
        all_tags.extend(tags)
    
    tag_counter = Counter(all_tags)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Top 30 tags bar chart
    top_30 = tag_counter.most_common(30)
    tags, counts = zip(*top_30)
    
    bars = ax1.barh(range(len(tags)), counts, color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(tags)))
    ax1.set_yticklabels(tags, fontsize=9)
    ax1.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Top 30 Most Frequent Tags', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(count + 20, i, str(count), va='center', fontsize=8)
    
    # Frequency distribution histogram
    frequencies = list(tag_counter.values())
    ax2.hist(frequencies, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Tag Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Tags', fontsize=12, fontweight='bold')
    ax2.set_title('Tag Frequency Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(np.median(frequencies), color='red', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(frequencies):.0f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rule_metrics_comparison(rules_df, figsize=(15, 5)):
    """
    Compare distributions of confidence, lift, and support
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = [
        ('confidence', 'Confidence', 'skyblue'),
        ('lift', 'Lift', 'lightcoral'),
        ('support', 'Support', 'lightgreen')
    ]
    
    for ax, (metric, label, color) in zip(axes, metrics):
        # Histogram
        n, bins, patches = ax.hist(rules_df[metric], bins=30, 
                                   color=color, edgecolor='black', alpha=0.7)
        
        # Add statistics
        mean_val = rules_df[metric].mean()
        median_val = rules_df[metric].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='blue', linestyle='--', 
                  linewidth=2, label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_recommendation_comparison(baseline_recs, enhanced_recs, figsize=(12, 6)):
    """
    Compare baseline vs enhanced recommendations
    """
    if not baseline_recs or not enhanced_recs:
        print("Need both baseline and enhanced recommendations")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data
    baseline_titles = [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] 
                       for r in baseline_recs[:10]]
    baseline_scores = [r['score'] for r in baseline_recs[:10]]
    
    enhanced_titles = [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] 
                       for r in enhanced_recs[:10]]
    enhanced_scores = [r['score'] for r in enhanced_recs[:10]]
    
    # Calculate common y-axis limit (max score + 10% padding)
    max_score = max(max(baseline_scores), max(enhanced_scores))
    y_limit = max_score * 1.1
    
    # Baseline recommendations
    bars1 = ax1.barh(range(len(baseline_titles)), baseline_scores, 
                     color='lightblue', edgecolor='black')
    ax1.set_yticks(range(len(baseline_titles)))
    ax1.set_yticklabels(baseline_titles, fontsize=9)
    ax1.set_xlabel('Overlap Score', fontsize=11, fontweight='bold')
    ax1.set_title('Baseline: Simple Tag Overlap', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, y_limit)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Enhanced recommendations
    bars2 = ax2.barh(range(len(enhanced_titles)), enhanced_scores, 
                     color='lightcoral', edgecolor='black')
    ax2.set_yticks(range(len(enhanced_titles)))
    ax2.set_yticklabels(enhanced_titles, fontsize=9)
    ax2.set_xlabel('Enhanced Score', fontsize=11, fontweight='bold')
    ax2.set_title('Enhanced: Rule-Based + Diversity', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, y_limit)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_diversity_analysis(recommendations, transactions_df, figsize=(12, 5)):
    """
    Analyze tag diversity in recommendations
    """
    if not recommendations:
        print("Need recommendations to analyze")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Tag diversity per recommendation
    tag_counts = []
    unique_tags = set()
    
    for rec in recommendations[:15]:
        movie = transactions_df[transactions_df['title'] == rec['title']]
        if not movie.empty:
            movie_tags = movie.iloc[0]['tags']
            tag_counts.append(len(movie_tags))
            unique_tags.update(movie_tags)
    
    # Plot 1: Tags per recommendation
    ax1.bar(range(len(tag_counts)), tag_counts, color='teal', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Recommendation Rank', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Tags', fontsize=11, fontweight='bold')
    ax1.set_title('Tag Count per Recommendation', fontsize=13, fontweight='bold')
    ax1.axhline(np.mean(tag_counts), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(tag_counts):.1f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Cumulative unique tags
    cumulative_unique = []
    seen_tags = set()
    
    for rec in recommendations[:15]:
        movie = transactions_df[transactions_df['title'] == rec['title']]
        if not movie.empty:
            movie_tags = set(movie.iloc[0]['tags'])
            seen_tags.update(movie_tags)
            cumulative_unique.append(len(seen_tags))
    
    ax2.plot(range(len(cumulative_unique)), cumulative_unique, 
             marker='o', linewidth=2, markersize=8, color='purple')
    ax2.fill_between(range(len(cumulative_unique)), cumulative_unique, alpha=0.3)
    ax2.set_xlabel('Recommendations Considered', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Unique Tags', fontsize=11, fontweight='bold')
    ax2.set_title('Tag Diversity Growth', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_all_visualizations(rules_df, transactions_df, baseline_recs=None, 
                             enhanced_recs=None, save_dir='visualizations',
                             network_top_n=200, heatmap_top_n=50):
    """
    Generate all visualizations and optionally save them
    
    Parameters:
    -----------
    network_top_n : int, default=200
        Number of top rules to display in network graph
    heatmap_top_n : int, default=50
        Number of top tags to display in co-occurrence heatmap
    """
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    figures = {}
    
    # 1. Rule scatter plot
    print("  1. Rule scatter plot...")
    figures['rule_scatter'] = plot_rule_scatter(rules_df)
    if save_dir:
        figures['rule_scatter'].savefig(f'{save_dir}/rule_scatter.png', dpi=300, bbox_inches='tight')
    
    # 2. Network graph
    print(f"  2. Rules network graph (top {network_top_n})...")
    figures['rules_network'] = plot_top_rules_network(rules_df, top_n=network_top_n)
    if save_dir:
        figures['rules_network'].savefig(f'{save_dir}/rules_network.png', dpi=300, bbox_inches='tight')
    
    # 3. Co-occurrence heatmap
    print(f"  3. Tag co-occurrence heatmap (top {heatmap_top_n})...")
    figures['cooccurrence'] = plot_tag_cooccurrence_heatmap(transactions_df, top_n=heatmap_top_n)
    if save_dir:
        figures['cooccurrence'].savefig(f'{save_dir}/tag_cooccurrence.png', dpi=300, bbox_inches='tight')
    
    # 4. Tag frequency
    print("  4. Tag frequency distribution...")
    figures['frequency'] = plot_tag_frequency_distribution(transactions_df)
    if save_dir:
        figures['frequency'].savefig(f'{save_dir}/tag_frequency.png', dpi=300, bbox_inches='tight')
    
    # 5. Metrics comparison
    print("  5. Rule metrics comparison...")
    figures['metrics'] = plot_rule_metrics_comparison(rules_df)
    if save_dir:
        figures['metrics'].savefig(f'{save_dir}/rule_metrics.png', dpi=300, bbox_inches='tight')
    
    # 6. Recommendation comparison (if provided)
    if baseline_recs and enhanced_recs:
        print("  6. Recommendation comparison...")
        figures['rec_comparison'] = plot_recommendation_comparison(baseline_recs, enhanced_recs)
        if save_dir:
            figures['rec_comparison'].savefig(f'{save_dir}/recommendation_comparison.png', 
                                             dpi=300, bbox_inches='tight')
        
        print("  7. Diversity analysis...")
        figures['diversity'] = plot_diversity_analysis(enhanced_recs, transactions_df)
        if save_dir:
            figures['diversity'].savefig(f'{save_dir}/diversity_analysis.png', 
                                        dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Generated {len(figures)} visualizations")
    if save_dir:
        print(f"📁 Saved to: {save_dir}/")
    
    return figures
