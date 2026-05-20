export interface ProjectRequest {
  description: string
  project_name?: string
  author?: string
  version?: string
  priority?: number
  backend_framework?: string
  frontend_framework?: string
  generate_tests?: boolean
  include_docker?: boolean
  include_docs?: boolean
  include_cicd?: boolean
  create_github_repo?: boolean
  github_token?: string
  github_private?: boolean
  tags?: string[]
}

export interface Project {
  id: string
  description: string
  project_name?: string
  author?: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  created_at: string
  completed_at?: string
  priority?: number
  result?: {
    project_dir: string
    backend_dir: string
    frontend_dir: string
    metadata?: Record<string, unknown>
  }
  error?: string
}

export interface GeneratorStatus {
  is_running: boolean
  queue_size: number
  processed_count: number
  failed_count: number
  current_project?: string
}

export interface QueueItem {
  id: string
  description: string
  project_name?: string
  priority: number
  created_at: string
  status: string
}

export interface QueueResponse {
  queue_size: number
  queue: QueueItem[]
}

export interface Stats {
  total_projects: number
  completed_projects: number
  failed_projects: number
  queue_size: number
  average_generation_time?: number
  projects_by_type?: Record<string, number>
  projects_by_framework?: Record<string, number>
}

export interface WebSocketMessage {
  type: string
  data: unknown
  timestamp: string
}

