import { apiService } from '@/services/api';
import type { JobSearchParams, JobSwipeAction, JobApplication } from '../types';
import type { Job } from '@/types';

export class JobService {
  async searchJobs(userId: string, params: JobSearchParams) {
    const response = await apiService.searchJobs(userId, params);
    if (!response.data) {
      throw new Error(response.error || 'Failed to search jobs');
    }
    return response.data;
  }

  async swipeJob(userId: string, action: JobSwipeAction) {
    const response = await apiService.swipeJob(userId, {
      job_id: action.jobId,
      action: action.action,
    });
    if (!response.data) {
      throw new Error(response.error || 'Failed to swipe job');
    }
    return response.data;
  }

  async applyToJob(userId: string, application: JobApplication) {
    const response = await apiService.applyToJob(userId, application.jobId, application.coverLetter);
    if (!response.data) {
      throw new Error(response.error || 'Failed to apply to job');
    }
    return response.data;
  }

  async getSavedJobs(userId: string): Promise<Job[]> {
    const response = await apiService.getSavedJobs(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get saved jobs');
    }
    return response.data;
  }

  async getLikedJobs(userId: string): Promise<Job[]> {
    const response = await apiService.getLikedJobs(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get liked jobs');
    }
    return response.data;
  }

  async getMatches(userId: string): Promise<Job[]> {
    const response = await apiService.getMatches(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get matches');
    }
    return response.data;
  }
}

export const jobService = new JobService();


