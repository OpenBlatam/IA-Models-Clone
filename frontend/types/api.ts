export interface DocumentRequest {
  query: string;
  business_area?: string;
  document_type?: string;
  priority?: number;
  metadata?: Record<string, any>;
  user_id?: string;
  session_id?: string;
}

export interface DocumentResponse {
  task_id: string;
  status: string;
  message: string;
  estimated_time?: number;
  queue_position?: number;
  created_at: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  result?: {
    content: string;
    format: string;
    word_count: number;
    generated_at: string;
    using_bul_system: boolean;
  };
  error?: string;
  created_at: string;
  updated_at: string;
  processing_time?: number;
}

export interface TaskListItem {
  task_id: string;
  status: string;
  progress: number;
  created_at: string;
  updated_at: string;
  user_id?: string;
  query_preview: string;
}

export interface TaskListResponse {
  tasks: TaskListItem[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface DocumentListItem {
  task_id: string;
  created_at: string;
  query_preview: string;
  business_area?: string;
  document_type?: string;
}

export interface DocumentsResponse {
  documents: DocumentListItem[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  uptime: string;
  active_tasks: number;
  total_requests: number;
  version: string;
}

export interface StatsResponse {
  total_requests: number;
  active_tasks: number;
  completed_tasks: number;
  success_rate: number;
  average_processing_time: number;
  uptime: string;
}

export interface WebSocketMessage {
  type: 'task_update' | 'initial_state' | 'connected' | 'error' | 'ping' | 'pong';
  task_id?: string;
  data?: any;
  message?: string;
  timestamp: string;
}


