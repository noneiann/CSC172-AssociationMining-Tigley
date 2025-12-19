"""
Enhanced Recommendation System with Diversity Metrics
"""
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_diversity_score(recommendations, all_tags):
    """
    Calculate diversity of recommended movies based on tag variety
    Higher score = more diverse recommendations
    """
    if not recommendations:
        return 0
    
    all_rec_tags = []
    for rec in recommendations:
        all_rec_tags.extend(rec.get('matching_tags', []))
    
    # Diversity = unique tags / total tags
    unique_tags = len(set(all_rec_tags))
    total_tags = len(all_rec_tags)
    
    return unique_tags / total_tags if total_tags > 0 else 0


def calculate_novelty_score(recommendations, tag_frequencies):
    """
    Calculate novelty - prefer recommendations with rare/unusual tags
    """
    if not recommendations:
        return 0
    
    novelty_scores = []
    for rec in recommendations:
        tags = rec.get('matching_tags', [])
        if tags:
            # Lower frequency = higher novelty
            avg_freq = np.mean([tag_frequencies.get(tag, 0) for tag in tags])
            novelty = 1 / (1 + avg_freq)  # Normalize
            novelty_scores.append(novelty)
    
    return np.mean(novelty_scores) if novelty_scores else 0


def hybrid_recommendation(seed_movie, rules_df, transactions_df, top_n=10, 
                         min_confidence=0.6, diversity_weight=0.3):
    """
    Enhanced recommendation with diversity and novelty weighting
    """
    seed_movie_data = transactions_df[transactions_df['title'] == seed_movie]
    if seed_movie_data.empty:
        print(f"Movie '{seed_movie}' not found")
        return None
    
    seed_tags = set(seed_movie_data.iloc[0]['tags'])
    print(f"Seed movie: {seed_movie}")
    print(f"Seed tags: {', '.join(list(seed_tags)[:8])}\n")
    
    # Calculate tag frequencies for novelty scoring
    all_tags = [tag for tags in transactions_df['tags'] for tag in tags]
    from collections import Counter
    tag_freq = Counter(all_tags)
    
    # Get rule-based recommendations
    filtered_rules = rules_df[rules_df['confidence'] >= min_confidence]
    
    recommended_tags = defaultdict(lambda: {'lifts': [], 'confidences': [], 'count': 0})
    
    for _, rule in filtered_rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        if antecedents.issubset(seed_tags):
            for tag in consequents:
                if tag not in seed_tags:
                    recommended_tags[tag]['lifts'].append(rule['lift'])
                    recommended_tags[tag]['confidences'].append(rule['confidence'])
                    recommended_tags[tag]['count'] += 1
    
    if not recommended_tags:
        print("⚠️  No recommendations found")
        return []
    
    # Score candidates
    candidates = []
    for _, movie in transactions_df.iterrows():
        if movie['title'] == seed_movie:
            continue
        
        movie_tags = set(movie['tags'])
        overlap = set(recommended_tags.keys()).intersection(movie_tags)
        
        if overlap:
            # Base score from rules
            rule_score = sum(
                sum(recommended_tags[tag]['lifts']) / len(recommended_tags[tag]['lifts']) 
                * recommended_tags[tag]['count']
                for tag in overlap
            )
            
            # Novelty bonus (prefer rare tags)
            novelty = np.mean([1 / (1 + tag_freq[tag]) for tag in overlap])
            
            # Diversity bonus (prefer movies with different genres)
            non_overlap = movie_tags - seed_tags
            diversity = len(non_overlap) / len(movie_tags) if movie_tags else 0
            
            # Combined score
            final_score = rule_score * (1 + diversity_weight * (novelty + diversity))
            
            candidates.append({
                'title': movie['title'],
                'score': final_score,
                'rule_score': rule_score,
                'novelty': novelty,
                'diversity': diversity,
                'matching_tags': overlap,
                'tag_count': len(overlap)
            })
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Display results
    print(f"=== Top {top_n} Enhanced Recommendations ===")
    for i, rec in enumerate(candidates[:top_n], 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Score: {rec['score']:.2f} (rule:{rec['rule_score']:.1f}, "
              f"novelty:{rec['novelty']:.2f}, diversity:{rec['diversity']:.2f})")
        print(f"   Tags ({rec['tag_count']}): {', '.join(list(rec['matching_tags'])[:5])}")
    
    return candidates[:top_n]


def compare_recommendation_quality(baseline_recs, enhanced_recs, tag_frequencies):
    """
    Compare baseline vs enhanced recommendations
    """
    print("\n" + "="*80)
    print("RECOMMENDATION QUALITY COMPARISON")
    print("="*80)
    
    if not baseline_recs or not enhanced_recs:
        print("Cannot compare - missing recommendations")
        return
    
    # Calculate metrics
    baseline_diversity = calculate_diversity_score(baseline_recs, tag_frequencies)
    enhanced_diversity = calculate_diversity_score(enhanced_recs, tag_frequencies)
    
    baseline_novelty = calculate_novelty_score(baseline_recs, tag_frequencies)
    enhanced_novelty = calculate_novelty_score(enhanced_recs, tag_frequencies)
    
    # Overlap
    baseline_titles = {r['title'] for r in baseline_recs}
    enhanced_titles = {r['title'] for r in enhanced_recs}
    overlap = len(baseline_titles & enhanced_titles)
    unique_enhanced = len(enhanced_titles - baseline_titles)
    
    print(f"\nDiversity Score:")
    print(f"  Baseline: {baseline_diversity:.3f}")
    print(f"  Enhanced: {enhanced_diversity:.3f}")
    print(f"  Improvement: {(enhanced_diversity - baseline_diversity):.3f} "
          f"({((enhanced_diversity/baseline_diversity - 1) * 100):.1f}%)")
    
    print(f"\nNovelty Score:")
    print(f"  Baseline: {baseline_novelty:.3f}")
    print(f"  Enhanced: {enhanced_novelty:.3f}")
    print(f"  Improvement: {(enhanced_novelty - baseline_novelty):.3f}")
    
    print(f"\nUniqueness:")
    print(f"  Overlap: {overlap}/{len(enhanced_recs)} movies")
    print(f"  New discoveries: {unique_enhanced}/{len(enhanced_recs)} movies")
    
    if unique_enhanced > 0:
        new_movies = list(enhanced_titles - baseline_titles)[:3]
        print(f"  Example new recommendations: {', '.join(new_movies)}")
