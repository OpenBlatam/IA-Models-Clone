import { apiService } from '@/services/api';
import type { Roadmap, Step } from '@/types';

export class RoadmapService {
  async getRoadmap(userId: string): Promise<Roadmap> {
    const response = await apiService.getRoadmap(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get roadmap');
    }
    return response.data;
  }

  async getProgress(userId: string): Promise<Roadmap> {
    const response = await apiService.getStepsProgress(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get progress');
    }
    return response.data;
  }

  async startStep(userId: string, stepId: string): Promise<Step> {
    const response = await apiService.startStep(userId, stepId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to start step');
    }
    return response.data;
  }

  async completeStep(userId: string, stepId: string, notes?: string): Promise<Step> {
    const response = await apiService.completeStep(userId, stepId, notes);
    if (!response.data) {
      throw new Error(response.error || 'Failed to complete step');
    }
    return response.data;
  }
}

export const roadmapService = new RoadmapService();


