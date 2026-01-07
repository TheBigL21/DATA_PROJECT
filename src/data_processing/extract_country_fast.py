"""
FAST COUNTRY EXTRACTION

Only extracts country for movies we actually have in our database.
Much faster than processing the entire title.akas file.
"""

from pathlib import Path
import pandas as pd
import gzip
from collections import defaultdict

# Priority countries for film production
PRIORITY_COUNTRIES = {
    'US', 'UK', 'FR', 'DE', 'IT', 'ES', 'JP', 'KR', 'CN', 'HK',
    'IN', 'CA', 'AU', 'BR', 'MX', 'SE', 'DK', 'NO', 'NL', 'BE'
}

# Region code to country name mapping
REGION_MAP = {
    'US': 'USA', 'GB': 'United Kingdom', 'UK': 'United Kingdom',
    'FR': 'France', 'DE': 'Germany', 'IT': 'Italy', 'ES': 'Spain',
    'NL': 'Netherlands', 'SE': 'Sweden', 'DK': 'Denmark', 'NO': 'Norway',
    'PL': 'Poland', 'BE': 'Belgium', 'AT': 'Austria', 'CH': 'Switzerland',
    'IE': 'Ireland', 'PT': 'Portugal', 'CZ': 'Czech Republic',
    'JP': 'Japan', 'KR': 'South Korea', 'CN': 'China', 'HK': 'Hong Kong',
    'IN': 'India', 'TH': 'Thailand', 'SG': 'Singapore', 'MY': 'Malaysia',
    'ID': 'Indonesia', 'PH': 'Philippines', 'TW': 'Taiwan', 'VN': 'Vietnam',
    'MX': 'Mexico', 'BR': 'Brazil', 'AR': 'Argentina', 'CL': 'Chile',
    'CO': 'Colombia', 'PE': 'Peru', 'VE': 'Venezuela',
    'AU': 'Australia', 'NZ': 'New Zealand',
    'ZA': 'South Africa', 'EG': 'Egypt', 'NG': 'Nigeria', 'CA': 'Canada',
}


def main(imdb_raw_dir: Path, output_dir: Path):
    """
    Fast country extraction - only for movies we have
    """
    akas_path = imdb_raw_dir / 'title.akas.tsv.gz'
    movies_core_path = output_dir / 'movies_core.parquet'

    if not akas_path.exists():
        raise FileNotFoundError(f"title.akas.tsv.gz not found at {akas_path}")

    if not movies_core_path.exists():
        raise FileNotFoundError(f"movies_core.parquet not found at {movies_core_path}")

    # Load tconsts we actually care about
    print("Loading existing movies...")
    movies_core = pd.read_parquet(movies_core_path)
    target_tconsts = set(movies_core['tconst'].values)
    print(f"  Looking for {len(target_tconsts):,} movies")

    # Extract countries only for movies we have
    print(f"\nProcessing {akas_path}...")
    movie_regions = defaultdict(list)  # tconst -> [(region, is_original), ...]

    found_count = 0
    with gzip.open(akas_path, 'rt', encoding='utf-8') as f:
        # Skip header
        header = f.readline()

        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(f"  Processed {i/1000000:.1f}M lines, found {found_count:,} relevant movies...")

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            titleId = parts[0]

            # SKIP if not in our target set
            if titleId not in target_tconsts:
                continue

            region = parts[3]
            isOriginalTitle = parts[7]

            # Skip if no region
            if region == '\\N' or not region:
                continue

            is_original = (isOriginalTitle == '1')
            movie_regions[titleId].append((region, is_original))

            if titleId not in movie_regions or len(movie_regions[titleId]) == 1:
                found_count += 1

    print(f"\nFound region data for {len(movie_regions):,} / {len(target_tconsts):,} movies")

    # Determine primary country for each movie
    print("Determining primary countries...")
    country_data = []

    for tconst in target_tconsts:
        if tconst not in movie_regions:
            # No region data found
            country_data.append({'tconst': tconst, 'country': 'Unknown'})
            continue

        regions_list = movie_regions[tconst]

        # Strategy 1: Use original title region if available
        original_regions = [r for r, is_orig in regions_list if is_orig]
        if original_regions:
            # Prefer priority countries
            priority_originals = [r for r in original_regions if r in PRIORITY_COUNTRIES]
            region_code = priority_originals[0] if priority_originals else original_regions[0]
        else:
            # Strategy 2: Use most common region, with priority weighting
            from collections import Counter
            region_counts = Counter([r for r, _ in regions_list])

            # Apply priority weighting
            weighted_regions = []
            for region, count in region_counts.items():
                weight = count * 2 if region in PRIORITY_COUNTRIES else count
                weighted_regions.append((region, weight))

            weighted_regions.sort(key=lambda x: x[1], reverse=True)
            region_code = weighted_regions[0][0]

        # Map to full country name
        country_name = REGION_MAP.get(region_code, 'Unknown')
        country_data.append({'tconst': tconst, 'country': country_name})

    df = pd.DataFrame(country_data)

    # Show distribution
    print("\nCountry distribution (top 20):")
    print(df['country'].value_counts().head(20))
    print(f"\nMovies with 'Unknown' country: {(df['country'] == 'Unknown').sum():,}")

    # Save
    output_path = output_dir / 'country_mapping.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved country mapping to {output_path}")
    print(f"Total movies: {len(df):,}")

    return df


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python extract_country_fast.py <imdb_raw_dir> <output_dir>")
        print("Example: python extract_country_fast.py ../../imdb_raw ../../output")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    main(raw_dir, out_dir)
