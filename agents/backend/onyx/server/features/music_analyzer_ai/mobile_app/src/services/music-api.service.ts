import { trackService } from './track/track.service';
import { analysisService } from './analysis/analysis.service';
import { recommendationsService } from './recommendations/recommendations.service';
import { discoveryService } from './discovery/discovery.service';
import { trendsService } from './trends/trends.service';
import { artistService } from './artist/artist.service';
import { historyService } from './history/history.service';
import { healthService } from './health/health.service';
import type {
  TrackSearchRequest,
  TrackSearchResponse,
  TrackAnalysis,
  TrackComparison,
  RecommendationsResponse,
  HealthCheck,
  Track,
  AudioFeatures,
  AudioAnalysis,
  ArtistComparison,
  ArtistEvolution,
  TrendsResponse,
  HistoryResponse,
  HistoryStats,
  ExportResponse,
} from '../types/api';
import type {
  AnalyzeTrackRequest,
  CompareTracksRequest,
} from './analysis/analysis.service';
import type { ContextualRecommendationRequest } from '../types/api';
import type { ArtistComparisonRequest } from '../types/api';

export class MusicApiService {
  get tracks() {
    return trackService;
  }

  get analysis() {
    return analysisService;
  }

  get recommendations() {
    return recommendationsService;
  }

  get discovery() {
    return discoveryService;
  }

  get trends() {
    return trendsService;
  }

  get artists() {
    return artistService;
  }

  get history() {
    return historyService;
  }

  get health() {
    return healthService;
  }

  async searchTracks(
    request: TrackSearchRequest
  ): Promise<TrackSearchResponse> {
    return this.tracks.searchTracks(request);
  }

  async analyzeTrack(request: AnalyzeTrackRequest): Promise<TrackAnalysis> {
    return this.analysis.analyzeTrack(request);
  }

  async analyzeTrackById(
    trackId: string,
    includeCoaching = true
  ): Promise<TrackAnalysis> {
    return this.analysis.analyzeTrackById(trackId, includeCoaching);
  }

  async getTrackInfo(trackId: string): Promise<Track> {
    return this.tracks.getTrackInfo(trackId);
  }

  async getAudioFeatures(trackId: string): Promise<AudioFeatures> {
    return this.tracks.getAudioFeatures(trackId);
  }

  async getRecommendations(
    trackId: string
  ): Promise<RecommendationsResponse> {
    return this.recommendations.getRecommendations(trackId);
  }

  async compareTracks(
    request: CompareTracksRequest
  ): Promise<TrackComparison> {
    return this.analysis.compareTracks(request);
  }

  async healthCheck(): Promise<HealthCheck> {
    return this.health.healthCheck();
  }

  async getCoaching(request: AnalyzeTrackRequest): Promise<TrackAnalysis> {
    return this.analysis.getCoaching(request);
  }

  async getAudioAnalysis(trackId: string): Promise<AudioAnalysis> {
    return this.tracks.getAudioAnalysis(trackId);
  }

  async getHistory(limit = 50): Promise<HistoryResponse> {
    return this.history.getHistory(limit);
  }

  async getHistoryStats(): Promise<HistoryStats> {
    return this.history.getHistoryStats();
  }

  async exportAnalysis(
    trackId: string,
    format: 'json' | 'text' | 'markdown' = 'json',
    includeCoaching = true
  ): Promise<ExportResponse> {
    return this.analysis.exportAnalysis(trackId, format, includeCoaching);
  }

  async getContextualRecommendations(
    request: ContextualRecommendationRequest
  ): Promise<RecommendationsResponse> {
    return this.recommendations.getContextualRecommendations(request);
  }

  async getTimeOfDayRecommendations(
    timeOfDay: string,
    limit = 20
  ): Promise<RecommendationsResponse> {
    return this.recommendations.getTimeOfDayRecommendations(timeOfDay, limit);
  }

  async getActivityRecommendations(
    activity: string,
    limit = 20
  ): Promise<RecommendationsResponse> {
    return this.recommendations.getActivityRecommendations(activity, limit);
  }

  async getMoodRecommendations(
    mood: string,
    limit = 20
  ): Promise<RecommendationsResponse> {
    return this.recommendations.getMoodRecommendations(mood, limit);
  }

  async discoverSimilarArtists(
    artistId: string,
    limit = 20
  ): Promise<TrackSearchResponse> {
    return this.discovery.discoverSimilarArtists(artistId, limit);
  }

  async discoverUnderground(limit = 20): Promise<TrackSearchResponse> {
    return this.discovery.discoverUnderground(limit);
  }

  async discoverMoodTransition(
    fromMood: string,
    toMood: string,
    limit = 20
  ): Promise<TrackSearchResponse> {
    return this.discovery.discoverMoodTransition(fromMood, toMood, limit);
  }

  async discoverFresh(limit = 20): Promise<TrackSearchResponse> {
    return this.discovery.discoverFresh(limit);
  }

  async compareArtists(request: ArtistComparisonRequest): Promise<ArtistComparison> {
    return this.artists.compareArtists(request);
  }

  async getArtistEvolution(artistId: string): Promise<ArtistEvolution> {
    return this.artists.getArtistEvolution(artistId);
  }

  async getTrendsPopularity(
    timeRange: 'short_term' | 'medium_term' | 'long_term' = 'medium_term',
    limit = 20
  ): Promise<TrendsResponse> {
    return this.trends.getTrendsPopularity(timeRange, limit);
  }

  async getTrendsArtists(
    timeRange: 'short_term' | 'medium_term' | 'long_term' = 'medium_term',
    limit = 20
  ): Promise<TrendsResponse> {
    return this.trends.getTrendsArtists(timeRange, limit);
  }
}

export const musicApiService = new MusicApiService();

export type {
  AnalyzeTrackRequest,
  CompareTracksRequest,
} from './analysis/analysis.service';
export type {
  ContextualRecommendationRequest,
  ArtistComparisonRequest,
  ExportResponse,
} from '../types/api';

