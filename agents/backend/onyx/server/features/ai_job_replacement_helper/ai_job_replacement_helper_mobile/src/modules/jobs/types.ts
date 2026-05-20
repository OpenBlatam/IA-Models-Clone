import type { Job } from '@/types';

export interface JobSearchParams {
  keywords?: string;
  location?: string;
  experience_level?: string;
  job_type?: string;
  limit?: number;
}

export interface JobSwipeAction {
  jobId: string;
  action: 'like' | 'dislike' | 'save';
}

export interface JobApplication {
  jobId: string;
  coverLetter?: string;
}

export interface JobFilters {
  keywords: string;
  location: string;
  salaryRange?: {
    min: number;
    max: number;
  };
  jobType?: string[];
  experienceLevel?: string[];
}

export interface JobCardProps {
  job: Job;
  onSwipe: (action: JobSwipeAction) => void;
  onApply: (jobId: string) => void;
}


