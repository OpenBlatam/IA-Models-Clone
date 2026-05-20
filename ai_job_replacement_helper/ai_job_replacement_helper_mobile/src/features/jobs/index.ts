// Domain
export { JobService } from './domain/job-service';
export type { JobSearchParams, JobSwipeAction, JobApplication, JobFilters } from './domain/job-types';

// Data
export { JobRepository } from './data/job-repository';

// Presentation
export { JobCard } from './presentation/components/job-card';
export { useJobSearch, useJobSwipe, useJobApplication } from './presentation/hooks/use-job-search';

// Constants
export { CARD_WIDTH, SWIPE_THRESHOLD, JOB_ACTIONS, JOB_FILTERS } from '@/modules/jobs/constants';

