import {
  searchQuerySchema,
  trackIdSchema,
  trackIdsSchema,
  paginationSchema,
  searchRequestSchema,
  analyzeTrackRequestSchema,
  compareTracksRequestSchema,
  userIdSchema,
  playlistNameSchema,
  moodSchema,
  activitySchema,
  timeOfDaySchema,
  ratingSchema,
  commentContentSchema,
  noteContentSchema,
  tagSchema,
  tagsSchema,
  exportFormatSchema,
  addToFavoritesRequestSchema,
  recommendationsRequestSchema,
  moodRecommendationsRequestSchema,
  activityRecommendationsRequestSchema,
  timeRecommendationsRequestSchema,
} from '@/lib/validations/music';

describe('Music Validation Schemas', () => {
  describe('searchQuerySchema', () => {
    it('should validate valid search query', () => {
      const result = searchQuerySchema.safeParse('test query');
      expect(result.success).toBe(true);
    });

    it('should reject empty query', () => {
      const result = searchQuerySchema.safeParse('');
      expect(result.success).toBe(false);
    });

    it('should reject query that is too short', () => {
      const result = searchQuerySchema.safeParse('a');
      expect(result.success).toBe(false);
    });
  });

  describe('trackIdSchema', () => {
    it('should validate valid track ID', () => {
      const result = trackIdSchema.safeParse('track-123');
      expect(result.success).toBe(true);
    });

    it('should reject empty track ID', () => {
      const result = trackIdSchema.safeParse('');
      expect(result.success).toBe(false);
    });
  });

  describe('trackIdsSchema', () => {
    it('should validate valid track IDs array', () => {
      const result = trackIdsSchema.safeParse(['track-1', 'track-2']);
      expect(result.success).toBe(true);
    });

    it('should reject empty array', () => {
      const result = trackIdsSchema.safeParse([]);
      expect(result.success).toBe(false);
    });

    it('should reject array with too many tracks', () => {
      const manyTracks = Array(11).fill('track-id');
      const result = trackIdsSchema.safeParse(manyTracks);
      expect(result.success).toBe(false);
    });
  });

  describe('paginationSchema', () => {
    it('should validate valid pagination', () => {
      const result = paginationSchema.safeParse({ page: 1, limit: 20 });
      expect(result.success).toBe(true);
    });

    it('should use default values', () => {
      const result = paginationSchema.safeParse({});
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(1);
        expect(result.data.limit).toBe(20);
      }
    });

    it('should reject negative page', () => {
      const result = paginationSchema.safeParse({ page: -1 });
      expect(result.success).toBe(false);
    });

    it('should reject limit over 100', () => {
      const result = paginationSchema.safeParse({ limit: 101 });
      expect(result.success).toBe(false);
    });
  });

  describe('searchRequestSchema', () => {
    it('should validate valid search request', () => {
      const result = searchRequestSchema.safeParse({
        query: 'test query',
        limit: 10,
      });
      expect(result.success).toBe(true);
    });

    it('should use default limit', () => {
      const result = searchRequestSchema.safeParse({ query: 'test' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.limit).toBe(10);
      }
    });
  });

  describe('analyzeTrackRequestSchema', () => {
    it('should validate with trackId', () => {
      const result = analyzeTrackRequestSchema.safeParse({
        trackId: 'track-123',
      });
      expect(result.success).toBe(true);
    });

    it('should validate with trackName', () => {
      const result = analyzeTrackRequestSchema.safeParse({
        trackName: 'Track Name',
      });
      expect(result.success).toBe(true);
    });

    it('should reject when neither trackId nor trackName provided', () => {
      const result = analyzeTrackRequestSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    it('should use default includeCoaching', () => {
      const result = analyzeTrackRequestSchema.safeParse({
        trackId: 'track-123',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.includeCoaching).toBe(true);
      }
    });
  });

  describe('compareTracksRequestSchema', () => {
    it('should validate valid compare request', () => {
      const result = compareTracksRequestSchema.safeParse({
        trackIds: ['track-1', 'track-2'],
      });
      expect(result.success).toBe(true);
    });

    it('should use default comparisonType', () => {
      const result = compareTracksRequestSchema.safeParse({
        trackIds: ['track-1', 'track-2'],
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.comparisonType).toBe('all');
      }
    });
  });

  describe('userIdSchema', () => {
    it('should validate valid user ID', () => {
      const result = userIdSchema.safeParse('user-123');
      expect(result.success).toBe(true);
    });

    it('should reject empty user ID', () => {
      const result = userIdSchema.safeParse('');
      expect(result.success).toBe(false);
    });

    it('should reject user ID that is too long', () => {
      const longId = 'a'.repeat(101);
      const result = userIdSchema.safeParse(longId);
      expect(result.success).toBe(false);
    });
  });

  describe('playlistNameSchema', () => {
    it('should validate valid playlist name', () => {
      const result = playlistNameSchema.safeParse('My Playlist');
      expect(result.success).toBe(true);
    });

    it('should reject empty playlist name', () => {
      const result = playlistNameSchema.safeParse('');
      expect(result.success).toBe(false);
    });
  });

  describe('moodSchema', () => {
    it('should validate valid mood', () => {
      const validMoods = ['happy', 'sad', 'energetic', 'calm'];
      validMoods.forEach((mood) => {
        const result = moodSchema.safeParse(mood);
        expect(result.success).toBe(true);
      });
    });

    it('should reject invalid mood', () => {
      const result = moodSchema.safeParse('invalid-mood');
      expect(result.success).toBe(false);
    });
  });

  describe('activitySchema', () => {
    it('should validate valid activity', () => {
      const validActivities = ['workout', 'study', 'party', 'relax'];
      validActivities.forEach((activity) => {
        const result = activitySchema.safeParse(activity);
        expect(result.success).toBe(true);
      });
    });

    it('should reject invalid activity', () => {
      const result = activitySchema.safeParse('invalid-activity');
      expect(result.success).toBe(false);
    });
  });

  describe('timeOfDaySchema', () => {
    it('should validate valid time of day', () => {
      const validTimes = ['morning', 'afternoon', 'evening', 'night'];
      validTimes.forEach((time) => {
        const result = timeOfDaySchema.safeParse(time);
        expect(result.success).toBe(true);
      });
    });

    it('should reject invalid time of day', () => {
      const result = timeOfDaySchema.safeParse('invalid-time');
      expect(result.success).toBe(false);
    });
  });

  describe('ratingSchema', () => {
    it('should validate valid rating', () => {
      for (let i = 1; i <= 5; i++) {
        const result = ratingSchema.safeParse(i);
        expect(result.success).toBe(true);
      }
    });

    it('should reject rating below 1', () => {
      const result = ratingSchema.safeParse(0);
      expect(result.success).toBe(false);
    });

    it('should reject rating above 5', () => {
      const result = ratingSchema.safeParse(6);
      expect(result.success).toBe(false);
    });

    it('should reject non-integer rating', () => {
      const result = ratingSchema.safeParse(3.5);
      expect(result.success).toBe(false);
    });
  });

  describe('commentContentSchema', () => {
    it('should validate valid comment', () => {
      const result = commentContentSchema.safeParse('This is a comment');
      expect(result.success).toBe(true);
    });

    it('should reject empty comment', () => {
      const result = commentContentSchema.safeParse('');
      expect(result.success).toBe(false);
    });

    it('should reject comment that is too long', () => {
      const longComment = 'a'.repeat(1001);
      const result = commentContentSchema.safeParse(longComment);
      expect(result.success).toBe(false);
    });
  });

  describe('noteContentSchema', () => {
    it('should validate valid note', () => {
      const result = noteContentSchema.safeParse('This is a note');
      expect(result.success).toBe(true);
    });

    it('should reject note that is too long', () => {
      const longNote = 'a'.repeat(5001);
      const result = noteContentSchema.safeParse(longNote);
      expect(result.success).toBe(false);
    });
  });

  describe('tagSchema', () => {
    it('should validate valid tag', () => {
      const result = tagSchema.safeParse('rock');
      expect(result.success).toBe(true);
    });

    it('should reject tag that is too long', () => {
      const longTag = 'a'.repeat(51);
      const result = tagSchema.safeParse(longTag);
      expect(result.success).toBe(false);
    });
  });

  describe('tagsSchema', () => {
    it('should validate valid tags array', () => {
      const result = tagsSchema.safeParse(['rock', 'pop', 'jazz']);
      expect(result.success).toBe(true);
    });

    it('should reject array with too many tags', () => {
      const manyTags = Array(21).fill('tag');
      const result = tagsSchema.safeParse(manyTags);
      expect(result.success).toBe(false);
    });
  });

  describe('exportFormatSchema', () => {
    it('should validate valid export format', () => {
      const validFormats = ['json', 'text', 'markdown'];
      validFormats.forEach((format) => {
        const result = exportFormatSchema.safeParse(format);
        expect(result.success).toBe(true);
      });
    });

    it('should reject invalid export format', () => {
      const result = exportFormatSchema.safeParse('invalid');
      expect(result.success).toBe(false);
    });
  });

  describe('addToFavoritesRequestSchema', () => {
    it('should validate valid favorites request', () => {
      const result = addToFavoritesRequestSchema.safeParse({
        userId: 'user-123',
        trackId: 'track-456',
        trackName: 'Track Name',
        artists: ['Artist 1'],
      });
      expect(result.success).toBe(true);
    });

    it('should reject when artists array is empty', () => {
      const result = addToFavoritesRequestSchema.safeParse({
        userId: 'user-123',
        trackId: 'track-456',
        trackName: 'Track Name',
        artists: [],
      });
      expect(result.success).toBe(false);
    });
  });

  describe('recommendationsRequestSchema', () => {
    it('should validate valid recommendations request', () => {
      const result = recommendationsRequestSchema.safeParse({
        trackId: 'track-123',
        limit: 20,
      });
      expect(result.success).toBe(true);
    });

    it('should use default limit', () => {
      const result = recommendationsRequestSchema.safeParse({
        trackId: 'track-123',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.limit).toBe(20);
      }
    });
  });

  describe('moodRecommendationsRequestSchema', () => {
    it('should validate valid mood recommendations request', () => {
      const result = moodRecommendationsRequestSchema.safeParse({
        mood: 'happy',
        limit: 10,
      });
      expect(result.success).toBe(true);
    });
  });

  describe('activityRecommendationsRequestSchema', () => {
    it('should validate valid activity recommendations request', () => {
      const result = activityRecommendationsRequestSchema.safeParse({
        activity: 'workout',
        limit: 15,
      });
      expect(result.success).toBe(true);
    });
  });

  describe('timeRecommendationsRequestSchema', () => {
    it('should validate valid time recommendations request', () => {
      const result = timeRecommendationsRequestSchema.safeParse({
        timeOfDay: 'morning',
        limit: 20,
      });
      expect(result.success).toBe(true);
    });
  });
});

