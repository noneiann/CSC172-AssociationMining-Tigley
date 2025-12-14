# CSC 172 Association Rule Mining Project Proposal\

**Student:** Rey Iann V. Tigley, 2022-0224  
**Date:** 14/12/2025

## 1. Project Title

Mining the Cinematic DNA: Semantic Association Rule Learning on the MovieLens Dataset

## 2. Problem Statement

Standard Movie Recommendation systems often rely on broad genres (eg. action or comedy or romance) or collaborative filtering. These often fail to capture the nuances or specific sub elements of a film. A user who likes zombies might not like all horror movies, but specifically enjoys the post apocalyptic survival theme, even if variants or even just no zombies are included. The model addresses the lack of semantic regularity in the recommendations by mining hidden associations within the MovieLens dataset, uncovering deep thematic and semantic links that broad genres miss.

## 3. Objectives

- Transform continuous tag relevance scores into a binary transactional format suitable for Apriori/FP-Growth.
- Identify high-confidence and high-lift relationships between descriptive tags (e.g., finding that Tarantino-esque implies Non-linear timeline).
- Demonstrate how these rules can improve recommendation diversity by linking seemingly unrelated films through shared semantic traits.

## 4. Dataset Plan

- Source: [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) - 15 million relevance scores, 1,129 tags
- Structure: Transactional. Assign each tag with a high enough relevance score (>0.5 or >0.8)
- Acquisition: Publicly available zip file (approx. 250MB) via the GroupLens website.

## 5. Technical Approach

### Preprocessing Pipeline

- **Pivot**  
  Reshape `genome-scores.csv` so that:

  - Rows represent **Movies**
  - Columns represent **Tags**
  - Cell values correspond to tag relevance scores

- **Thresholding**  
  Convert continuous relevance scores (0.0–1.0) into binary Boolean values.  
  Planned thresholds to define tag “presence”:

  - `> 0.5`
  - `> 0.8`

- **Filtering**  
  Remove overly generic or non-informative **stop-tags** (e.g., _“good movie”_, _“classic”_) to reduce noise and improve rule quality.

### Model / Algorithm

- **Algorithm**
  - Apriori
    - Classic data mining algorithm for discovering frequent itemsets and association rule mining
  - FP-Growth
    - Preferred over Apriori due to better memory efficiency for large datasets

### Hardware

- **Google Colab** (High-RAM runtime recommended for faster processing)

## 6. Expected Challenges & Mitigations

### Challenge: The "Long Tail" of Sparsity

- **Issue**  
  Most tags occur very rarely, resulting in near-zero support for niche but potentially interesting concepts.
- **Mitigation**  
  Apply:
  - A **low minimum support threshold** (e.g., `0.01%`)
  - A **high minimum confidence threshold** (e.g., `≥ 70%`)  
    This helps capture rare yet strong association rules.

### Challenge: Synonyms and Redundancy

- **Issue**  
  Rules such as `{Aliens} → {Extraterrestrial}` are technically correct but offer little analytical value.
- **Mitigation**
  - Implement a **pre-processing step** to group or normalize synonymous tags, or
  - Manually filter out **trivial or redundant rules** during post-processing analysis.

### Challenge: Computational Cost

- **Issue**  
  With **1,129 unique tags**, the search space becomes extremely large
- **Mitigation**
  - Prune the dataset to the **top 200 most relevant tags** if memory constraints arise, and/or
  - Use the **FP-Growth algorithm**, which is significantly more efficient than Apriori for large, sparse datasets.
