import pandas as pd

# Configuration parameters
RELEVANCE_MIN = 0.4
RELEVANCE_MAX = 1.0

# Load the datasets
print("Loading datasets...")
df_movies = pd.read_csv("datasets/raw/movies.csv")
df_genome_scores = pd.read_csv("datasets/raw/genome-scores.csv")
df_genome_tags = pd.read_csv("datasets/raw/genome-tags.csv")

# Filter relevance scores
print(f"Original genome scores count: {len(df_genome_scores)}")
df_genome_scores = df_genome_scores[
    (df_genome_scores['relevance'] >= RELEVANCE_MIN) & 
    (df_genome_scores['relevance'] <= RELEVANCE_MAX)
]
print(f"Filtered genome scores count ({RELEVANCE_MIN}-{RELEVANCE_MAX}): {len(df_genome_scores)}")

# Define stop-tags 
STOP_TAGS = {
    'good movie', 'great movie', 'classic', 'favorite', 'excellent',
    'bad movie', 'boring', 'entertaining', 'must see', 'highly recommended',
    'good', 'great', 'bad', 'movie', 'film', 'cinema',
    'watchable', 'seen it', 'own it', 'loved it', 'liked it', 'hated it',
    'overrated', 'underrated', 'interesting', 'recommended', 'original', 'clever',
    'fun', 'awesome', 'terrible', 'waste of time', 'masterpiece',
    'fun movie', 'buddy movie', 'war movie', 'animal movie', 'road movie', 
    'stoner movie', 'girlie movie', 'funniest movies', 'best war films',
    'easily confused with other movie(s) (title)', 'movielens top pick',
    'dumb', 'dumb but funny', 'stupid', 'stupid as hell',
    'complicated plot', 'no plot', 'original plot', 'plot',
    'highly quotable', 'quotable',
    'afi 100', 'afi 100 (laughs)',
    'book was better', 'better than the original',
    'funny as hell', 'scary as hell', 'sad as hell',
    'slow paced', 'fast paced',
    'long', 'too long', 'too short',
    'travel',
    'awful', 'horrible', 'amazing cinematography', 'beautiful', 'stunning',
    'beautiful scenery', 'scenic', 'visually stunning',
    'big budget', 'low budget', 'pg', 'pg-13',
    'cute', 'cute!', 'predictable', 'goofy',
}

# Define synonym mappings 
TAG_SYNONYMS = {
    'france': 'french',
    'england': 'british',
    'world war ii': 'wwii',
    'world war i': 'wwi',
    'gay character': 'gay',
    'biography': 'biopic',
    'teenager': 'teen',
    'teenagers': 'teen',
    'teens': 'teen',
    'teen movie': 'teen',
    'adolescence': 'teen',
    'sci fi': 'sci-fi',
    'scifi': 'sci-fi',
    'intelligent sci-fi': 'sci-fi',
    'scifi cult': 'sci-fi',
    'super hero': 'superhero',
    'super-hero': 'superhero',
    'superheroes': 'superhero',
    'film noir': 'noir',
    'noir thriller': 'noir',
    'dystopic future': 'dystopia',
    'coming of age': 'coming-of-age',
    'post apocalyptic': 'post-apocalyptic',
    'android(s)/cyborg(s)': 'androids',
    'cyborgs': 'androids',
    'based on a book': 'based on book',
    'adapted from:book': 'based on book',
    'based on a comic': 'based on comic',
    'adapted from:comic': 'based on comic',
    'based on a true story': 'based on true story',
    'adapted from:game': 'based on video game',
    'based on a video game': 'based on video game',
    'based on a play': 'adapted from play',
    'based on a tv show': 'adapted from tv',
    'true story': 'based on true story',
    'fairy tale': 'fairy tales',
    'sequels': 'sequel',
    'animated': 'animation',
    'hilarious': 'comedy',
    'very funny': 'comedy',
    'funny': 'comedy',
    'love': 'romance',
    'love story': 'romance',
    'romantic': 'romance',
    'chick flick': 'romantic comedy',
    'action packed': 'action',
    'fight scenes': 'action',
    'scary': 'horror',
    'creepy': 'horror',
    'science fiction': 'sci-fi',
    'book': 'based on book',
    'books': 'based on book',
    'car chase': 'chase',
    'gory': 'gore',
    'bloody': 'gore',
    'blood': 'gore',
    'splatter': 'gore',
    'brutal': 'violence',
    'violent': 'violence',
    'brutality': 'violence',
    'gangsters': 'gangster',
    'mob': 'organized crime',
    'surreal': 'surrealism',
    'weird': 'surrealism',
    'dreamlike': 'surrealism',
    'suspenseful': 'suspense',
    'tense': 'suspense',
    'futuristic': 'future',
    'cartoon': 'animation',
    'computer animation': 'animation',
    'talking animals': 'animals',
    'historical': 'history',
    'us history': 'history',
    'sexual': 'sexuality',
    'pornography': 'sexuality',
    'murder mystery': 'murder',
    'police investigation': 'investigation',
    'comic': 'based on comic',
    'comics': 'based on comic',
    'comic book': 'based on comic',
    'comic book adaption': 'based on comic',
    'graphic novel': 'based on comic',
    'sex': 'sexuality',
    'sexual': 'sexuality',
    'sexy': 'sexuality',
    'dramatic': 'drama',
    'assassin': 'assassination',
    'children' : 'kids',
    'kids and family': 'family',
    'silly fun' : 'silly',
    'father son relationship': 'father-son relationship',
    'video games': 'based on video game',
    'videogame': 'based on video game',
    'computer game': 'based on video game',
    'dancing': 'dance',
    'lawyer': 'lawyers',
    'dragons': 'dragon',
    'monsters': 'monster',
    'aliens': 'alien',
    'robots': 'robot',
    'trains': 'train',
    'new york city': 'new york',
    'math': 'mathematics',
    'weed': 'marijuana',
    'coen bros': 'coen brothers',
    'political': 'politics',
    'world politics': 'politics',
    'philosophical': 'philosophy',
    'satirical': 'satire',
    'zombies': 'zombie',
    'vampires': 'vampire',
    'nazis': 'nazi',
    'werewolves': 'werewolf',
    'geeks': 'geek',
    'jews': 'jewish',
    'stop motion': 'stop-motion',
    'sword fight': 'sword fighting',
    'fast paced': 'fast-paced',
    'slow paced': 'slow-paced',
    'paranoid': 'paranoia',
    'male nudity': 'nudity',
    'notable nudity': 'nudity',
    'nudity (rear)': 'nudity',
    'nudity (topless)': 'nudity',
    'nudity (topless - brief)': 'nudity',
    'nudity (topless - notable)': 'nudity',
    'nudity (full frontal)': 'nudity',
    'nudity (full frontal - brief)': 'nudity',
    'nudity (full frontal - notable)': 'nudity',
    'japanese': 'japan',
    'tokyo': 'japan',
    'russian': 'russia',
    'irish': 'ireland',
    'irish accent': 'ireland',
    'berlin': 'germany',
    'east germany': 'germany',
    'mother son relationship': 'mother-son relationship',
    'father daughter relationship': 'father-daughter relationship',
    'mother daughter relationship': 'mother-daughter relationship',
    'artistic': 'art',
    'artist': 'art',
    'drug abuse': 'drug addiction',
    'heroin': 'drugs',
    'iraq war': 'iraq',
    'civil war': 'american civil war',
    'twist': 'plot twist',
    'twist ending': 'plot twist',
    'twists & turns': 'plot twist',
    'black comedy': 'dark comedy',
    'dark humor': 'dark comedy',
    'dialogue driven': 'dialogue',
    'entirely dialogue': 'dialogue',
    'catastrophe': 'disaster',
    'author:neil gaiman': 'neil gaiman',
    'bank robbery': 'robbery',
    'private detective': 'detective',
    'treasure hunt': 'treasure',
    'courtroom drama': 'courtroom',
    'fake documentary': 'documentary',
    'photography': 'photographer',
    'environment': 'environmental',
    'writers': 'writing',
    'remake': 'remade',
    'homophobia': 'gay',
    'homosexuality': 'gay',
    'queer': 'gay',
    'christian': 'christianity',
    'soccer': 'football',
    'cute!': 'cute',
    '80s': '1980s',
    '70s': '1970s',
    '60s': '1960s',
    '50s': '1950s',
    '40s': '1940s',
    '30s': '1930s',
    '20s': '1920s',
    'fantasy world': 'fantasy',
    'modern fantasy': 'fantasy',
    'dogs': 'dog',
    'horses': 'horse',
    'music business': 'music',
    'musicians': 'music',
    'bombs': 'explosions',
    'nuclear bomb': 'nuclear',
    'nuclear war': 'nuclear',
    'police corruption': 'corruption',
    'political corruption': 'corruption',
    'memory loss': 'memory',
    'short-term memory loss': 'memory',
    'beer': 'drinking',
    'ghosts/afterlife': 'afterlife',
    'ghosts': 'afterlife',
    'natural disaster': 'disaster',
    'australian': 'australia',
    'brazilian': 'brazil',
    'mexican': 'mexico',
    'argentinian': 'argentina',
    'school': 'high school',
    'addiction': 'drug addiction',
    'psychiatrist': 'psychiatry',
    'psychological': 'psychology',
    'neo-noir': 'noir',
    'jewish': 'judaism',
    'vietnam war': 'vietnam',
    'court': 'courtroom',
    'spy': 'spies',
    'spying': 'espionage',
    'suicide attempt': 'suicide',
    'goth': 'gothic',
    'cloning': 'clones',
    'conspiracy theory': 'conspiracy',
    'alcoholism': 'drinking',
    'inspiring': 'inspirational',
    'visuals': 'visual',
    'witches': 'witch',
    'weapons': 'guns',
    'airplane': 'aviation',
    'strange': 'bizarre',
    'non-linear': 'nonlinear',
    'mistaken identity': 'identity',
    'life philosophy': 'philosophy',
    'cult film': 'cult',
    'james bond': 'bond',
    '007 (series)': '007',
    'hacking': 'hackers',
    'humorous': 'humor',
    'screwball comedy': 'screwball',
    'race issues': 'race',
    'goretastic': 'gore',
    'gambling': 'casino',
    'alien invasion': 'alien',
    'hostage': 'kidnapping',
    'nostalgia': 'nostalgic',
    'underwater': 'ocean',
    'apocalypse': 'post-apocalyptic',
    'alternate universe': 'parallel universe',
    'mafia': 'gangster',
    'marriage': 'wedding',
    'life & death': 'death',
    'south africa': 'africa',
    'latin america': 'south america',
}

print("Cleaning data...")

# Merge genome scores with tags to enable filtering
df_genome = df_genome_scores.merge(df_genome_tags, on='tagId')

# Replace synonyms with canonical tags
print(f"Tags before synonym merging: {df_genome['tag'].nunique()}")
df_genome['tag'] = df_genome['tag'].replace(TAG_SYNONYMS)
print(f"Tags after synonym merging: {df_genome['tag'].nunique()}")

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

# Create cleaned tags table with synonym mappings applied
df_cleaned_tags = df_genome[['tagId', 'tag']].drop_duplicates()

# Save cleaned datasets
df_cleaned_scores.to_csv('datasets/cleaned/genome-scores-cleaned.csv', index=False)
df_cleaned_tags.to_csv('datasets/cleaned/genome-tags-cleaned.csv', index=False)

# Display summary
print("\nCleaned Data Summary:")
print(f"Total movie-tag pairs: {len(df_cleaned_scores)}")
print(f"Unique movies: {df_cleaned_scores['movieId'].nunique()}")
print(f"Unique tags: {df_cleaned_scores['tagId'].nunique()}")
print(f"\nRelevance range: {df_cleaned_scores['relevance'].min():.4f} - {df_cleaned_scores['relevance'].max():.4f}")
print(f"Mean relevance: {df_cleaned_scores['relevance'].mean():.4f}")

print("\nSaved cleaned data to:")
print("  - datasets/cleaned/genome-scores-cleaned.csv")
print("  - datasets/cleaned/genome-tags-cleaned.csv")

