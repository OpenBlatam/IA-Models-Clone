export enum ProjectStatus {
  QUEUED = 'queued',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface Project {
  project_id: string;
  project_name: string;
  description: string;
  author: string;
  status: ProjectStatus;
  created_at: string;
  updated_at?: string;
  project_dir?: string;
  metadata?: Record<string, any>;
}

export interface ProjectRequest {
  description: string;
  project_name?: string;
  author?: string;
  version?: string;
  priority?: number;
  backend_framework?: string;
  frontend_framework?: string;
  generate_tests?: boolean;
  include_docker?: boolean;
  include_docs?: boolean;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface ProjectResponse {
  project_id: string;
  status: string;
  message: string;
  project_info?: Record<string, any>;
}

export interface GenerationTask {
  task_id: string;
  project_request: ProjectRequest;
  status: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: Record<string, any>;
  error?: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  score?: number;
}

export interface QueueStatus {
  total: number;
  queued: number;
  processing: number;
  completed: number;
  failed: number;
}

export interface Stats {
  total_projects: number;
  completed_projects: number;
  failed_projects: number;
  average_generation_time?: number;
  uptime?: number;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}

