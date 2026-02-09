import { apiClient } from '@/utils/api-client';
import { API_ENDPOINTS } from '@/utils/config';
import type { MusicTrack } from '@/types/api';

export const musicService = {
  async listTracks(style?: string): Promise<{ tracks: MusicTrack[] }> {
    const params = style ? `?style=${style}` : '';
    return apiClient.get<{ tracks: MusicTrack[] }>(
      `${API_ENDPOINTS.MUSIC.TRACKS}${params}`
    );
  },
};


