import { apiClient } from '@/utils/api-client';
import { API_ENDPOINTS } from '@/utils/config';
import type {
  VideoGenerationRequest,
  VideoGenerationResponse,
  BatchGenerationRequest,
  BatchGenerationResponse,
} from '@/types/api';

export const videoService = {
  async generateVideo(
    request: VideoGenerationRequest
  ): Promise<VideoGenerationResponse> {
    return apiClient.post<VideoGenerationResponse>(
      API_ENDPOINTS.VIDEO.GENERATE,
      request
    );
  },

  async getVideoStatus(videoId: string): Promise<VideoGenerationResponse> {
    return apiClient.get<VideoGenerationResponse>(
      API_ENDPOINTS.VIDEO.STATUS(videoId)
    );
  },

  async downloadVideo(
    videoId: string,
    onProgress?: (progress: number) => void
  ): Promise<{ uri: string; headers: Record<string, string> }> {
    return apiClient.download(API_ENDPOINTS.VIDEO.DOWNLOAD(videoId), onProgress);
  },

  async deleteVideo(videoId: string): Promise<{ message: string }> {
    return apiClient.delete<{ message: string }>(
      API_ENDPOINTS.VIDEO.DELETE(videoId)
    );
  },

  async uploadScript(
    file: { uri: string; type: string; name: string },
    onProgress?: (progress: number) => void
  ): Promise<{ text: string; filename: string; size: number }> {
    return apiClient.upload<{ text: string; filename: string; size: number }>(
      API_ENDPOINTS.VIDEO.UPLOAD_SCRIPT,
      file,
      onProgress
    );
  },

  async registerWebhook(
    videoId: string,
    webhookUrl: string
  ): Promise<{ message: string }> {
    return apiClient.post<{ message: string }>(
      API_ENDPOINTS.VIDEO.WEBHOOK(videoId),
      { webhook_url: webhookUrl }
    );
  },

  async getVideoVersions(videoId: string): Promise<{
    video_id: string;
    versions: unknown[];
    total_versions: number;
  }> {
    return apiClient.get<{
      video_id: string;
      versions: unknown[];
      total_versions: number;
    }>(API_ENDPOINTS.VIDEO.VERSIONS(videoId));
  },

  async getVideoVersion(
    videoId: string,
    versionNumber: number
  ): Promise<unknown> {
    return apiClient.get<unknown>(
      API_ENDPOINTS.VIDEO.VERSION(videoId, versionNumber)
    );
  },

  async compareVersions(
    videoId: string,
    version1: number,
    version2: number
  ): Promise<unknown> {
    return apiClient.get<unknown>(
      `${API_ENDPOINTS.VIDEO.COMPARE_VERSIONS(videoId)}?version1=${version1}&version2=${version2}`
    );
  },

  async exportVideo(
    videoId: string,
    platforms: string[]
  ): Promise<{
    video_id: string;
    platforms: string[];
    status: string;
    message: string;
  }> {
    return apiClient.post<{
      video_id: string;
      platforms: string[];
      status: string;
      message: string;
    }>(API_ENDPOINTS.VIDEO.EXPORT(videoId), { platforms });
  },

  async addWatermark(
    videoId: string,
    options: {
      watermark_text?: string;
      watermark_image?: string;
      position?: string;
      opacity?: number;
      size?: number;
    }
  ): Promise<{
    video_id: string;
    status: string;
    message: string;
  }> {
    return apiClient.post<{
      video_id: string;
      status: string;
      message: string;
    }>(API_ENDPOINTS.VIDEO.WATERMARK(videoId), options);
  },

  async transcribeVideo(
    videoId: string,
    language?: string
  ): Promise<{ video_id: string; transcription: unknown }> {
    return apiClient.post<{ video_id: string; transcription: unknown }>(
      API_ENDPOINTS.VIDEO.TRANSCRIBE(videoId),
      { language }
    );
  },

  async addKenBurnsEffect(
    videoId: string,
    options: {
      zoom?: number;
      pan_x?: number;
      pan_y?: number;
    }
  ): Promise<{
    video_id: string;
    status: string;
    message: string;
  }> {
    return apiClient.post<{
      video_id: string;
      status: string;
      message: string;
    }>(API_ENDPOINTS.VIDEO.KEN_BURNS(videoId), options);
  },

  async shareVideo(
    videoId: string,
    options: {
      shared_with_email?: string;
      shared_with_id?: string;
      permission?: string;
      is_public?: boolean;
      expires_at?: string;
    }
  ): Promise<unknown> {
    return apiClient.post<unknown>(API_ENDPOINTS.VIDEO.SHARE(videoId), options);
  },

  async getVideoShares(videoId: string): Promise<{ shares: unknown[] }> {
    return apiClient.get<{ shares: unknown[] }>(
      API_ENDPOINTS.VIDEO.SHARES(videoId)
    );
  },

  async submitFeedback(
    videoId: string,
    feedback: {
      rating: number;
      comment?: string;
      tags?: string[];
    }
  ): Promise<unknown> {
    return apiClient.post<unknown>(
      API_ENDPOINTS.VIDEO.FEEDBACK(videoId),
      feedback
    );
  },

  async getVideoFeedback(videoId: string): Promise<{
    feedbacks: unknown[];
    statistics: unknown;
  }> {
    return apiClient.get<{
      feedbacks: unknown[];
      statistics: unknown;
    }>(API_ENDPOINTS.VIDEO.FEEDBACK(videoId));
  },

  async scheduleVideo(
    videoId: string,
    options: {
      scheduled_at: string;
      request: VideoGenerationRequest;
      timezone?: string;
      repeat?: string;
    }
  ): Promise<unknown> {
    return apiClient.post<unknown>(
      API_ENDPOINTS.VIDEO.SCHEDULE(videoId),
      options
    );
  },
};

export const batchService = {
  async generateBatch(
    request: BatchGenerationRequest
  ): Promise<BatchGenerationResponse> {
    return apiClient.post<BatchGenerationResponse>(
      API_ENDPOINTS.BATCH.GENERATE,
      request
    );
  },

  async getBatchStatus(videoIds: string[]): Promise<unknown> {
    const idsParam = videoIds.join(',');
    return apiClient.get<unknown>(
      `${API_ENDPOINTS.BATCH.STATUS}?video_ids=${idsParam}`
    );
  },
};


