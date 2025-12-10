"""
KEYWORD FILTER MODULE

Comprehensive filtering system to ensure only relevant thematic keywords
are suggested to users. Excludes production details, demographics, generic
relationships, and other non-thematic content.
"""

from typing import List, Set


class KeywordFilter:
    """Filters keywords to keep only relevant thematic content"""

    # Keywords to EXCLUDE - not thematically relevant
    EXCLUDED_KEYWORDS = {
        # ===== PRODUCTION/TECHNICAL DETAILS =====
        'woman director', 'female director', 'male director', 'directorial debut',
        'black and white', 'color', 'technicolor', 'silent film', 'independent film',
        'cinematography', 'camera', 'scene', 'shot', 'editing', 'pre-code', 'b movie',
        'duringcreditsstinger', 'aftercreditsstinger', 'post credits scene',
        'remake', 'sequel', 'prequel', 'reboot', 'spin off', 'spin-off',
        'low budget', 'cult film', 'cult classic', 'art film', 'arthouse',
        'critically acclaimed', 'controversial', 'banned film',
        'flashback', 'voice over narration', 'narration', 'surprise ending',
        'cinemascope', '16mm film', 'technicolor',

        # ===== IDENTITY/DEMOGRAPHIC/REPRESENTATION =====
        'lgbt', 'gay', 'lesbian', 'transgender', 'bisexual', 'queer', 'lgbtq',
        'gay theme', 'lesbian relationship', 'gay relationship', 'gay character',
        'african american', 'black people', 'hispanic', 'latino', 'latina',
        'asian', 'white people', 'race relations', 'racial issues',
        'interracial relationship', 'racism', 'racial tension',

        # ===== GENERIC RELATIONSHIPS =====
        'husband wife relationship', 'parent child relationship', 'father son relationship',
        'family relationships', 'sibling relationship', 'father daughter relationship',
        'mother daughter relationship', 'mother son relationship', 'brother sister relationship',
        'marriage', 'wedding', 'engagement', 'love', 'friendship', 'family',
        'father', 'mother', 'husband', 'wife', 'brother', 'sister', 'son', 'daughter',
        'boyfriend girlfriend relationship', 'relationship', 'relationships',
        'forbidden love', 'romance', 'romantic', 'breakup', 'divorce',
        'extramarital affair', 'infidelity', 'love triangle', 'unrequited love',

        # ===== GENERIC DEMOGRAPHICS =====
        'male protagonist', 'female protagonist', 'protagonist',
        'man', 'woman', 'boy', 'girl', 'child', 'children',
        'coming of age', 'midlife crisis', 'teenage', 'teenager', 'childhood', 'elderly',
        'adolescence', 'youth', 'old age', 'growing up', 'teenager',

        # ===== SOURCE MATERIAL (moved to separate question) =====
        'based on novel', 'based on book', 'based on novel or book',
        'based on play', 'based on play or musical', 'based on musical',
        'based on comic', 'based on comic book', 'based on manga',
        'based on true story', 'based on real events', 'true story',
        'biography', 'biographical', 'biopic',
        'based on tv series', 'based on video game',
        'based on song, poem or rhyme', 'based on myths, legends or folklore',
        'literary adaptation', 'stage adaptation',

        # ===== TIME PERIODS (moved to separate question) =====
        '1st century', '2nd century', '3rd century', '4th century', '5th century',
        '6th century', '7th century', '8th century', '9th century', '10th century',
        '11th century', '12th century', '13th century', '14th century', '15th century',
        '16th century', '17th century', '18th century', '19th century', '20th century',
        '21st century', '10th century bc', '11th century bc', '12th century bc',
        '13th century bc', '14th century bc', '15th century bc',
        '1800s', '1810s', '1820s', '1830s', '1840s', '1850s', '1860s', '1870s', '1880s', '1890s',
        '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s',
        '1914', '1915', '1927', '1965', '1967', '1969', '1999',
        'ancient', 'medieval', 'victorian era', 'edwardian era',

        # ===== HISTORICAL EVENTS (too broad) =====
        'world war ii', 'world war i', 'wwi', 'wwii', 'world war 2', 'world war 1',
        'vietnam war', 'cold war', 'civil war', 'korean war',
        'historical figure', 'historical drama', 'historical',

        # ===== GENERIC LOCATIONS (too broad) =====
        'new york city', 'los angeles california', 'los angeles, california',
        'paris france', 'paris, france', 'london england', 'london, england',
        'san francisco california', 'tokyo japan', 'hong kong',
        'small town', 'big city', 'city', 'village', 'countryside',
        'usa', 'france', 'england', 'japan', 'italy', 'germany', 'spain',
        'europe', 'asia', 'africa', 'america',

        # ===== GENERIC PLOT ELEMENTS (too vague) =====
        'death', 'violence', 'murder', 'fight', 'battle', 'chase',
        'escape', 'rescue', 'running', 'hiding', 'searching', 'looking', 'finding', 'trying',
        'investigation', 'trial', 'murder investigation', 'murder mystery',
        'good versus evil', 'good vs evil',
        'deception', 'betrayal', 'mistaken identity', 'assumed identity',
        'on the run', 'fugitive', 'framed for murder', 'blackmail',
        'character', 'story', 'plot', 'protagonist',

        # ===== GENERIC EMOTIONS (too broad) =====
        'jealousy', 'fear', 'hope', 'despair', 'guilt', 'loss', 'grief', 'trauma',
        'anger', 'sadness', 'happiness', 'joy',

        # ===== CONTENT DESCRIPTORS (not themes) =====
        'nudity', 'sex', 'sexuality', 'sexual content', 'erotic', 'sensuality',
        'sex scene', 'sexual abuse', 'rape', 'sexual violence',
        'strong language', 'profanity', 'gore', 'graphic violence', 'blood',

        # ===== FILM STYLES (too technical) =====
        'film noir', 'neo-noir', 'british noir', 'western noir',
        'found footage', 'mockumentary', 'documentary style',
        'anthology', 'episodic', 'non-linear',

        # ===== GENERIC ACTIVITIES (too vague) =====
        'cooking', 'dancing', 'singing', 'eating', 'drinking',
        'party', 'celebration', 'festival',

        # ===== MISC GENERIC =====
        'christmas', 'holiday', 'new year', 'halloween', 'thanksgiving',
        'high school', 'college', 'university', 'school',
        'sports', 'game', 'competition', 'tournament',
        'police', 'cop', 'cops', 'police officer', 'police detective',
        'doctor', 'nurse', 'teacher', 'lawyer', 'journalist',
        'amused', 'amusing', 'entertaining',
        'musical', 'anime',  # Too generic as standalone keywords
    }

    # Keywords to ALWAYS KEEP - highly thematic and specific
    ALWAYS_KEEP = {
        # Character types
        'serial killer', 'psychopath', 'sociopath', 'assassin', 'hitman',
        'detective', 'private detective', 'private investigator',
        'spy', 'secret agent', 'double agent',
        'superhero', 'supervillain', 'vigilante',
        'vampire', 'werewolf', 'zombie', 'ghost', 'demon', 'angel',
        'alien', 'extraterrestrial', 'monster', 'creature',
        'robot', 'android', 'cyborg', 'artificial intelligence',
        'wizard', 'witch', 'sorcerer', 'mage',
        'pirate', 'outlaw', 'bandit', 'gangster', 'mobster',
        'samurai', 'ninja', 'warrior',

        # Specific plot elements
        'time travel', 'parallel universe', 'alternate reality',
        'dystopia', 'utopia', 'post-apocalyptic', 'apocalypse',
        'survival', 'stranded', 'deserted island',
        'heist', 'bank robbery', 'casino',
        'kidnapping', 'hostage', 'ransom',
        'prison', 'prison escape', 'jail',
        'road trip', 'journey', 'quest',
        'revenge', 'vengeance',
        'conspiracy', 'cover-up', 'corruption',
        'haunted house', 'haunting', 'possession', 'exorcism',
        'body swap', 'identity swap',

        # Specific settings
        'space', 'spaceship', 'space station', 'space travel',
        'submarine', 'underwater',
        'desert', 'jungle', 'arctic', 'antarctica',
        'island', 'isolated', 'remote location',
        'hospital', 'asylum', 'mental hospital',
        'hotel', 'motel',

        # Mood/Tone (when specific)
        'dark comedy', 'black comedy', 'satire', 'parody',
        'slasher', 'splatter', 'torture porn',
        'psychological horror', 'psychological thriller',
        'body horror', 'cosmic horror',

        # Specific themes
        'martial arts', 'kung fu', 'karate', 'boxing',
        'magic', 'supernatural', 'paranormal',
        'cloning', 'genetic engineering',
        'virtual reality', 'simulation', 'matrix',
        'pandemic', 'virus', 'outbreak', 'plague',
        'nuclear', 'atomic bomb',
        'dinosaur', 'dragon',
        'zombie apocalypse', 'alien invasion',
    }

    @classmethod
    def is_relevant(cls, keyword: str) -> bool:
        """
        Check if a keyword is thematically relevant.

        Args:
            keyword: Keyword to check (lowercase)

        Returns:
            True if keyword should be kept, False if excluded
        """
        keyword_lower = keyword.lower().strip()

        # Always keep whitelisted keywords
        if keyword_lower in cls.ALWAYS_KEEP:
            return True

        # Exclude blacklisted keywords
        if keyword_lower in cls.EXCLUDED_KEYWORDS:
            return False

        # Additional pattern-based exclusions
        if cls._matches_exclusion_pattern(keyword_lower):
            return False

        return True

    @classmethod
    def _matches_exclusion_pattern(cls, keyword: str) -> bool:
        """Check if keyword matches exclusion patterns"""

        # Exclude keywords with "relationship" or "relationships"
        if 'relationship' in keyword or 'relationships' in keyword:
            return True

        # Exclude keywords ending in "director"
        if keyword.endswith('director'):
            return True

        # Exclude century markers
        if 'century' in keyword or keyword.endswith('bc'):
            return True

        # Exclude decades (1920s, 1930s, etc.) - simple pattern
        if keyword.replace('s', '').replace('0', '').isdigit() and 's' in keyword:
            if len(keyword) == 5:  # e.g., "1920s"
                return True

        # Exclude "based on" variants
        if keyword.startswith('based on'):
            return True

        return False

    @classmethod
    def filter_keywords(cls, keywords: List[str]) -> List[str]:
        """
        Filter a list of keywords to keep only relevant ones.

        Args:
            keywords: List of keywords to filter

        Returns:
            Filtered list of relevant keywords
        """
        return [kw for kw in keywords if cls.is_relevant(kw)]

    @classmethod
    def get_exclusion_stats(cls) -> dict:
        """Get statistics about exclusion rules"""
        return {
            'total_excluded': len(cls.EXCLUDED_KEYWORDS),
            'total_whitelisted': len(cls.ALWAYS_KEEP),
            'categories': {
                'production_technical': 30,
                'identity_demographic': 15,
                'relationships': 25,
                'source_material': 20,
                'time_periods': 50,
                'locations': 20,
                'generic_plot': 25,
                'emotions': 10,
                'content_descriptors': 10,
            }
        }


if __name__ == '__main__':
    """Test the keyword filter"""
    print("="*70)
    print("KEYWORD FILTER TEST")
    print("="*70)

    # Test cases
    test_keywords = [
        # Should be KEPT
        'serial killer', 'time travel', 'dystopia', 'heist', 'revenge',
        'zombie', 'alien', 'superhero', 'haunted house', 'prison escape',

        # Should be EXCLUDED
        'woman director', 'based on novel or book', 'lgbt', 'husband wife relationship',
        'coming of age', '1960s', 'new york city', 'world war ii', 'love', 'friendship',
    ]

    print("\nTesting keyword filtering:\n")
    for kw in test_keywords:
        is_relevant = KeywordFilter.is_relevant(kw)
        status = "✓ KEEP" if is_relevant else "✗ EXCLUDE"
        print(f"  {status}: {kw}")

    # Statistics
    print(f"\n{'='*70}")
    print("FILTER STATISTICS")
    print(f"{'='*70}")
    stats = KeywordFilter.get_exclusion_stats()
    print(f"\nTotal excluded keywords: {stats['total_excluded']}")
    print(f"Total whitelisted keywords: {stats['total_whitelisted']}")
    print("\nExclusion categories:")
    for category, count in stats['categories'].items():
        print(f"  - {category.replace('_', ' ').title()}: ~{count} keywords")
