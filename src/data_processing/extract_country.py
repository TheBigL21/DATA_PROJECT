"""
COUNTRY EXTRACTION MODULE

Purpose: Extract primary production country for each movie from title.akas.tsv.gz

Strategy:
- Use isOriginalTitle=1 entries first (original release country)
- Fall back to most common region for the movie
- Prioritize major film-producing countries (US, UK, FR, JP, etc.)

Output: country_mapping.parquet with (tconst, country)
"""

from pathlib import Path
import pandas as pd
import gzip
from collections import Counter


# Priority countries for film production (used for tie-breaking)
PRIORITY_COUNTRIES = {
    'US', 'UK', 'FR', 'DE', 'IT', 'ES', 'JP', 'KR', 'CN', 'HK',
    'IN', 'CA', 'AU', 'BR', 'MX', 'SE', 'DK', 'NO', 'NL', 'BE'
}


def extract_countries_from_akas(akas_path: Path, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Extract primary country for each movie from title.akas file.

    Strategy:
    1. First try isOriginalTitle=1 entries
    2. Then use most frequent region with priority weighting
    3. Map region codes to full country names

    Args:
        akas_path: Path to title.akas.tsv.gz
        chunk_size: Process in chunks to save memory

    Returns:
        DataFrame with (tconst, country)
    """
    print(f"Processing {akas_path}...")

    # Dictionary to store region data per movie
    movie_regions = {}  # tconst -> list of (region, is_original)

    # Read file in chunks to save memory
    with gzip.open(akas_path, 'rt', encoding='utf-8') as f:
        # Skip header
        header = f.readline().strip().split('\t')

        chunk = []
        for i, line in enumerate(f):
            if i % 500000 == 0:
                print(f"  Processed {i:,} lines...")

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            titleId = parts[0]
            region = parts[3]
            isOriginalTitle = parts[7]

            # Skip if no region
            if region == '\\N' or not region:
                continue

            # Store (region, is_original_flag)
            is_original = (isOriginalTitle == '1')

            if titleId not in movie_regions:
                movie_regions[titleId] = []
            movie_regions[titleId].append((region, is_original))

    print(f"  Found regions for {len(movie_regions):,} movies")

    # Determine primary country for each movie
    print("Determining primary countries...")
    country_data = []

    for tconst, regions_list in movie_regions.items():
        # Strategy 1: If there's an original title, use that region
        original_regions = [r for r, is_orig in regions_list if is_orig]
        if original_regions:
            # Prefer priority countries if multiple originals
            priority_originals = [r for r in original_regions if r in PRIORITY_COUNTRIES]
            if priority_originals:
                country = priority_originals[0]
            else:
                country = original_regions[0]
            country_data.append({'tconst': tconst, 'country': country})
            continue

        # Strategy 2: Use most common region, with priority weighting
        region_counts = Counter([r for r, _ in regions_list])

        # Apply priority weighting
        weighted_regions = []
        for region, count in region_counts.items():
            weight = count * 2 if region in PRIORITY_COUNTRIES else count
            weighted_regions.append((region, weight))

        # Sort by weight and pick highest
        weighted_regions.sort(key=lambda x: x[1], reverse=True)
        country = weighted_regions[0][0]
        country_data.append({'tconst': tconst, 'country': country})

    df = pd.DataFrame(country_data)
    print(f"Extracted countries for {len(df):,} movies")

    return df


def map_region_to_country(region_code: str) -> str:
    """
    Map 2-letter region code to full country name.

    This matches the naming in region_weighting.py
    """
    region_map = {
        'US': 'USA',
        'GB': 'United Kingdom',
        'UK': 'United Kingdom',
        'FR': 'France',
        'DE': 'Germany',
        'IT': 'Italy',
        'ES': 'Spain',
        'NL': 'Netherlands',
        'SE': 'Sweden',
        'DK': 'Denmark',
        'NO': 'Norway',
        'PL': 'Poland',
        'BE': 'Belgium',
        'AT': 'Austria',
        'CH': 'Switzerland',
        'IE': 'Ireland',
        'PT': 'Portugal',
        'CZ': 'Czech Republic',
        'JP': 'Japan',
        'KR': 'South Korea',
        'CN': 'China',
        'HK': 'Hong Kong',
        'IN': 'India',
        'TH': 'Thailand',
        'SG': 'Singapore',
        'MY': 'Malaysia',
        'ID': 'Indonesia',
        'PH': 'Philippines',
        'TW': 'Taiwan',
        'VN': 'Vietnam',
        'MX': 'Mexico',
        'BR': 'Brazil',
        'AR': 'Argentina',
        'CL': 'Chile',
        'CO': 'Colombia',
        'PE': 'Peru',
        'VE': 'Venezuela',
        'AU': 'Australia',
        'NZ': 'New Zealand',
        'ZA': 'South Africa',
        'EG': 'Egypt',
        'NG': 'Nigeria',
        'CA': 'Canada',
    }

    return region_map.get(region_code, 'Unknown')


def main(imdb_raw_dir: Path, output_dir: Path):
    """
    Main execution: Extract country data from title.akas.tsv.gz

    Args:
        imdb_raw_dir: Directory containing title.akas.tsv.gz
        output_dir: Directory to save country_mapping.parquet
    """
    akas_path = imdb_raw_dir / 'title.akas.tsv.gz'

    if not akas_path.exists():
        raise FileNotFoundError(f"title.akas.tsv.gz not found at {akas_path}")

    # Extract countries
    countries_df = extract_countries_from_akas(akas_path)

    # Map region codes to full country names
    print("\nMapping region codes to country names...")
    countries_df['country'] = countries_df['country'].apply(map_region_to_country)

    # Show distribution
    print("\nCountry distribution (top 20):")
    print(countries_df['country'].value_counts().head(20))

    # Save to parquet
    output_path = output_dir / 'country_mapping.parquet'
    countries_df.to_parquet(output_path, index=False)
    print(f"\nSaved country mapping to {output_path}")
    print(f"Total movies with country data: {len(countries_df):,}")

    return countries_df


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python extract_country.py <imdb_raw_dir> <output_dir>")
        print("Example: python extract_country.py ../../imdb_raw ../../output")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    main(raw_dir, out_dir)
