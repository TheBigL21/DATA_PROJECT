// Import API types for mapping
import { ApiMovie } from './api';

export interface Movie {
  id: string;
  title: string;
  year: number;
  rating: number;
  genres: string[];
  keywords: string[];
  description: string;
  poster: string;
  director: string;
  runtime: number;
}

export const GENRES = [
  'Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 
  'Thriller', 'Romance', 'Adventure', 'Animation', 'Crime'
] as const;

export const KEYWORDS = [
  'Revenge', 'Love', 'Adventure', 'Mystery', 'Heist',
  'Espionage', 'Survival', 'Friendship', 'Family', 'War'
] as const;

export const ERAS = [
  { id: 'classic', label: 'Classic', range: [1920, 1979] },
  { id: '80s-90s', label: '80s & 90s', range: [1980, 1999] },
  { id: '2000s', label: '2000s', range: [2000, 2014] },
  { id: 'modern', label: 'Modern', range: [2015, 2025] },
  { id: 'any', label: 'Any Era', range: [1920, 2025] },
] as const;

// Sample movie database for demonstration
export const MOVIES: Movie[] = [
  {
    id: '1',
    title: 'Inception',
    year: 2010,
    rating: 8.8,
    genres: ['Action', 'Sci-Fi', 'Thriller'],
    keywords: ['Heist', 'Mystery', 'Adventure'],
    description: 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
    poster: 'https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Ber.jpg',
    director: 'Christopher Nolan',
    runtime: 148,
  },
  {
    id: '2',
    title: 'The Dark Knight',
    year: 2008,
    rating: 9.0,
    genres: ['Action', 'Crime', 'Drama'],
    keywords: ['Revenge', 'Mystery', 'Survival'],
    description: 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
    poster: 'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
    director: 'Christopher Nolan',
    runtime: 152,
  },
  {
    id: '3',
    title: 'Parasite',
    year: 2019,
    rating: 8.5,
    genres: ['Drama', 'Thriller', 'Comedy'],
    keywords: ['Family', 'Survival', 'Mystery'],
    description: 'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.',
    poster: 'https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg',
    director: 'Bong Joon Ho',
    runtime: 132,
  },
  {
    id: '4',
    title: 'Interstellar',
    year: 2014,
    rating: 8.6,
    genres: ['Adventure', 'Drama', 'Sci-Fi'],
    keywords: ['Survival', 'Love', 'Family'],
    description: 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
    poster: 'https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
    director: 'Christopher Nolan',
    runtime: 169,
  },
  {
    id: '5',
    title: 'Pulp Fiction',
    year: 1994,
    rating: 8.9,
    genres: ['Crime', 'Drama'],
    keywords: ['Revenge', 'Adventure', 'Friendship'],
    description: 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
    poster: 'https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg',
    director: 'Quentin Tarantino',
    runtime: 154,
  },
  {
    id: '6',
    title: 'The Shawshank Redemption',
    year: 1994,
    rating: 9.3,
    genres: ['Drama'],
    keywords: ['Friendship', 'Survival', 'Adventure'],
    description: 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
    poster: 'https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg',
    director: 'Frank Darabont',
    runtime: 142,
  },
  {
    id: '7',
    title: 'The Matrix',
    year: 1999,
    rating: 8.7,
    genres: ['Action', 'Sci-Fi'],
    keywords: ['Adventure', 'Survival', 'Mystery'],
    description: 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
    poster: 'https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg',
    director: 'The Wachowskis',
    runtime: 136,
  },
  {
    id: '8',
    title: 'Fight Club',
    year: 1999,
    rating: 8.8,
    genres: ['Drama', 'Thriller'],
    keywords: ['Revenge', 'Friendship', 'Mystery'],
    description: 'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.',
    poster: 'https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg',
    director: 'David Fincher',
    runtime: 139,
  },
  {
    id: '9',
    title: 'Goodfellas',
    year: 1990,
    rating: 8.7,
    genres: ['Crime', 'Drama'],
    keywords: ['Family', 'Friendship', 'Survival'],
    description: 'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners.',
    poster: 'https://image.tmdb.org/t/p/w500/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg',
    director: 'Martin Scorsese',
    runtime: 146,
  },
  {
    id: '10',
    title: 'Se7en',
    year: 1995,
    rating: 8.6,
    genres: ['Crime', 'Drama', 'Thriller'],
    keywords: ['Mystery', 'Revenge', 'Survival'],
    description: 'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.',
    poster: 'https://image.tmdb.org/t/p/w500/6yoghtyTpznpBik8EngEmJskVUO.jpg',
    director: 'David Fincher',
    runtime: 127,
  },
  {
    id: '11',
    title: 'Spirited Away',
    year: 2001,
    rating: 8.6,
    genres: ['Animation', 'Adventure', 'Drama'],
    keywords: ['Adventure', 'Family', 'Friendship'],
    description: 'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.',
    poster: 'https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg',
    director: 'Hayao Miyazaki',
    runtime: 125,
  },
  {
    id: '12',
    title: 'The Godfather',
    year: 1972,
    rating: 9.2,
    genres: ['Crime', 'Drama'],
    keywords: ['Family', 'Revenge', 'Survival'],
    description: 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son.',
    poster: 'https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
    director: 'Francis Ford Coppola',
    runtime: 175,
  },
  {
    id: '13',
    title: 'Blade Runner 2049',
    year: 2017,
    rating: 8.0,
    genres: ['Sci-Fi', 'Drama', 'Thriller'],
    keywords: ['Mystery', 'Adventure', 'Survival'],
    description: 'Young Blade Runner K\'s discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard.',
    poster: 'https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg',
    director: 'Denis Villeneuve',
    runtime: 164,
  },
  {
    id: '14',
    title: 'Whiplash',
    year: 2014,
    rating: 8.5,
    genres: ['Drama'],
    keywords: ['Survival', 'Revenge', 'Adventure'],
    description: 'A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student\'s potential.',
    poster: 'https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg',
    director: 'Damien Chazelle',
    runtime: 106,
  },
  {
    id: '15',
    title: 'Get Out',
    year: 2017,
    rating: 7.7,
    genres: ['Horror', 'Thriller'],
    keywords: ['Mystery', 'Survival', 'Family'],
    description: 'A young African-American visits his white girlfriend\'s parents for the weekend, where his simmering uneasiness about their reception of him eventually reaches a boiling point.',
    poster: 'https://image.tmdb.org/t/p/w500/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg',
    director: 'Jordan Peele',
    runtime: 104,
  },
  {
    id: '16',
    title: 'La La Land',
    year: 2016,
    rating: 8.0,
    genres: ['Comedy', 'Drama', 'Romance'],
    keywords: ['Love', 'Adventure', 'Friendship'],
    description: 'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.',
    poster: 'https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg',
    director: 'Damien Chazelle',
    runtime: 128,
  },
  {
    id: '17',
    title: 'Mad Max: Fury Road',
    year: 2015,
    rating: 8.1,
    genres: ['Action', 'Adventure', 'Sci-Fi'],
    keywords: ['Survival', 'Adventure', 'Revenge'],
    description: 'In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners, a psychotic worshiper, and a drifter named Max.',
    poster: 'https://image.tmdb.org/t/p/w500/8tZYtuWezp8JbcsvHYO0O46tFbo.jpg',
    director: 'George Miller',
    runtime: 120,
  },
  {
    id: '18',
    title: 'The Grand Budapest Hotel',
    year: 2014,
    rating: 8.1,
    genres: ['Adventure', 'Comedy', 'Crime'],
    keywords: ['Adventure', 'Friendship', 'Mystery'],
    description: 'A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel\'s glorious years under an exceptional concierge.',
    poster: 'https://image.tmdb.org/t/p/w500/eWdyYQreja6JGCzqHWXpWHDrrPo.jpg',
    director: 'Wes Anderson',
    runtime: 99,
  },
  {
    id: '19',
    title: 'John Wick',
    year: 2014,
    rating: 7.4,
    genres: ['Action', 'Crime', 'Thriller'],
    keywords: ['Revenge', 'Survival', 'Adventure'],
    description: 'An ex-hit-man comes out of retirement to track down the gangsters that killed his dog and took everything from him.',
    poster: 'https://image.tmdb.org/t/p/w500/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg',
    director: 'Chad Stahelski',
    runtime: 101,
  },
  {
    id: '20',
    title: 'Dune',
    year: 2021,
    rating: 8.0,
    genres: ['Sci-Fi', 'Adventure', 'Drama'],
    keywords: ['Adventure', 'Survival', 'Family'],
    description: 'Feature adaptation of Frank Herbert\'s science fiction novel about the son of a noble family entrusted with the protection of the most valuable asset in the galaxy.',
    poster: 'https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg',
    director: 'Denis Villeneuve',
    runtime: 155,
  },
];

export type QuestionnaireAnswers = {
  evening_type: string;  // Frontend ID: 'chill_evening' | 'date_night' | 'family_night' | 'friends_night'
  genres: string[];       // 1-2 genres
  keywords: string[];     // 0-3 keywords/themes
  era: string;            // Direct era ID: 'new_era' | 'golden_era' | 'millennium' | 'old_school' | 'any'
  source_material?: string;  // Optional: 'book' | 'true_story' | 'comic' | 'play_musical' | 'original' | 'any'
};

export type SwipeFeedback = {
  movieId: string;
  satisfaction: number; // 0 = dislike, 0.3 = like, 1.0 = love
};

/**
 * Maps backend API movie format to frontend Movie format
 */
export function mapApiMovieToMovie(apiMovie: ApiMovie): Movie {
  return {
    id: String(apiMovie.movie_id),
    title: apiMovie.title,
    year: apiMovie.year,
    rating: apiMovie.combined_rating || apiMovie.avg_rating,
    genres: apiMovie.genres,
    keywords: apiMovie.keywords || [],
    description: apiMovie.description || '',
    poster: apiMovie.poster_url || '/placeholder.svg',
    director: apiMovie.director,
    runtime: apiMovie.runtime,
  };
}

/**
 * Map frontend evening type ID to backend string format
 */
export const EVENING_TYPE_MAP: Record<string, string> = {
  'chill_evening': 'Chill Evening by myself',
  'date_night': 'Date night',
  'family_night': 'Family night',
  'friends_night': 'Friends night'
};
