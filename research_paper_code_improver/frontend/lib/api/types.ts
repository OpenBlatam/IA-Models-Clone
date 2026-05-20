export interface PaperResponse {
  source: string
  title: string
  authors: string[]
  abstract: string
  sections_count: number
  content_length: number
  metadata: Record<string, unknown>
}

export interface PaperLinkRequest {
  url: string
  download?: boolean
}

export interface TrainingRequest {
  paper_ids?: string[]
  model_name?: string
  epochs?: number
  use_all_papers?: boolean
}

export interface TrainingResponse {
  model_id: string
  status: string
  papers_count: number
  training_examples: number
  epochs: number
  model_path: string
}

export interface CodeImproveRequest {
  github_repo: string
  file_path: string
  branch?: string
  model_id?: string
}

export interface CodeImproveResponse {
  original_code: string
  improved_code: string
  suggestions: Array<{
    type: string
    description: string
    line?: number
    severity?: string
  }>
  repo: string
  file_path: string
  improvements_applied: number
}

export interface RepositoryAnalyzeRequest {
  github_repo: string
  branch?: string
  model_id?: string
  max_files?: number
}

export interface RepositoryAnalyzeResponse {
  repo: string
  files_analyzed: number
  total_improvements: number
  improvements: Array<{
    file_path: string
    suggestions: Array<{
      type: string
      description: string
      line?: number
    }>
  }>
}

export interface ModelStatusResponse {
  model_id: string
  status: string
  config?: Record<string, unknown>
  error?: string
}

export interface HealthCheckResponse {
  status: string
  service: string
  version: string
  vector_store: {
    papers_indexed: number
    available: boolean
  }
  paper_storage: {
    total_papers: number
  }
  features: {
    rag: boolean
    cache: boolean
    analyzer: boolean
  }
}

export interface Paper {
  id: string
  title: string
  authors: string[]
  abstract: string
  source: string
  sections_count: number
  content_length: number
  metadata: Record<string, unknown>
}

export interface PapersListResponse {
  papers: Paper[]
  total: number
  statistics: {
    total_papers: number
    total_size: number
  }
}

export interface VectorStoreStats {
  papers_indexed: number
  collection_name: string
}

export interface CacheStats {
  total_entries: number
  hit_rate: number
  size_mb: number
}

export interface MetricsStats {
  requests: {
    total: number
    successful: number
    failed: number
  }
  improvements: {
    total: number
    average_time_ms: number
  }
  papers: {
    uploaded: number
    processed: number
  }
}




