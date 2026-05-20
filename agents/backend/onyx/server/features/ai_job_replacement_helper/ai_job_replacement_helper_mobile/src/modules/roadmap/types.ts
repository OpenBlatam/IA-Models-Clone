import type { Step, Roadmap } from '@/types';

export interface StepProgress {
  stepId: string;
  status: 'not_started' | 'in_progress' | 'completed';
  startedAt?: Date;
  completedAt?: Date;
  notes?: string;
}

export interface RoadmapFilters {
  category?: string;
  status?: StepProgress['status'];
  searchQuery?: string;
}

export interface StepResource {
  type: 'article' | 'video' | 'tool' | 'template';
  title: string;
  url: string;
  description?: string;
}


