import { JobRepository } from '../data/job-repository';
import type { JobSearchParams, JobSwipeAction, JobApplication } from './job-types';
import type { Job, JobSwipeResponse } from '@/types';

export class JobService {
  constructor(private repository: JobRepository = new JobRepository()) {}

  async searchJobs(userId: string, params: JobSearchParams): Promise<{ jobs: Job[]; total: number }> {
    return this.repository.searchJobs(userId, params);
  }

  async swipeJob(userId: string, action: JobSwipeAction): Promise<JobSwipeResponse> {
    return this.repository.swipeJob(userId, action);
  }

  async applyToJob(userId: string, application: JobApplication): Promise<{ success: boolean; message: string }> {
    return this.repository.applyToJob(userId, application);
  }

  async getSavedJobs(userId: string): Promise<Job[]> {
    return this.repository.getSavedJobs(userId);
  }

  async getLikedJobs(userId: string): Promise<Job[]> {
    return this.repository.getLikedJobs(userId);
  }

  async getMatches(userId: string): Promise<Job[]> {
    return this.repository.getMatches(userId);
  }
}

