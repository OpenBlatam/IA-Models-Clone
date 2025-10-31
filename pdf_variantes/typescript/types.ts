/**
 * PDF Variantes API - TypeScript Types
 * Generated from Python Pydantic models
 * Version: 2.0.0
 */

// Enums
export enum VariantStatus {
  PENDING = "pending",
  GENERATING = "generating",
  COMPLETED = "completed",
  FAILED = "failed",
  STOPPED = "stopped",
  ULTRA_FAST = "ultra_fast",
  GPU_ACCELERATED = "gpu_accelerated",
  PARALLEL_PROCESSING = "parallel_processing",
  CACHED = "cached",
  STREAMING = "streaming",
}

export enum PDFProcessingStatus {
  UPLOADING = "uploading",
  PROCESSING = "processing",
  READY = "ready",
  EDITING = "editing",
  ERROR = "error",
  OPTIMIZING = "optimizing",
  CACHING = "caching",
  GPU_PROCESSING = "gpu_processing",
  PARALLEL_PROCESSING = "parallel_processing",
  STREAMING = "streaming",
}

export enum TopicCategory {
  MAIN = "main",
  SUPPORTING = "supporting",
  RELATED = "related",
  CROSS_REFERENCE = "cross_reference",
  AI_GENERATED = "ai_generated",
  SEMANTIC = "semantic",
  CONTEXTUAL = "contextual",
  DYNAMIC = "dynamic",
}

// Core Types
export interface PDFMetadata {
  file_id: string;
  original_filename: string;
  file_size: number;
  upload_date: string; // ISO datetime string
  page_count: number;
  word_count: number;
  language?: string;
  performance_level?: string;
  extreme_speed_level?: string;
  assembly_optimization?: string;
  cache_strategy?: string;
  processing_mode?: string;
  performance_metrics?: PerformanceMetrics;
  cached_at?: string;
  optimization_applied?: boolean;
  assembly_boost_enabled?: boolean;
  lock_free_processing?: boolean;
  zero_copy_enabled?: boolean;
}

export interface PerformanceMetrics {
  processing_time?: number;
  throughput?: number;
  memory_usage?: number;
  cpu_usage?: number;
  cache_hit_rate?: number;
  [key: string]: unknown;
}

export interface EditedPage {
  page_number: number;
  content: string;
  modifications: Record<string, unknown>[];
  edited_at: string;
}

export interface PDFDocument {
  id: string;
  metadata: PDFMetadata;
  status: PDFProcessingStatus;
  original_content: string;
  processed_content?: string;
  extracted_text?: string;
  edited_pages?: EditedPage[];
  variants?: PDFVariant[];
  topics?: TopicItem[];
  created_at: string;
  updated_at: string;
  user_id?: string;
}

export interface VariantConfiguration {
  creativity_level?: number;
  similarity_threshold?: number;
  max_variations?: number;
  preserve_structure?: boolean;
  custom_prompt?: string;
  [key: string]: unknown;
}

export interface PDFVariant {
  id: string;
  document_id: string;
  variant_number: number;
  content: string;
  title?: string;
  description?: string;
  status: VariantStatus;
  similarity_score?: number;
  quality_score?: number;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface TopicItem {
  id: string;
  document_id: string;
  topic: string;
  category: TopicCategory;
  relevance_score: number;
  frequency: number;
  related_topics?: string[];
  extracted_at: string;
  metadata?: Record<string, unknown>;
}

export interface BrainstormIdea {
  id: string;
  document_id: string;
  idea: string;
  category?: string;
  relevance_score?: number;
  feasibility_score?: number;
  created_at: string;
  metadata?: Record<string, unknown>;
}

// Request/Response Types
export interface PDFUploadRequest {
  filename?: string;
  auto_process?: boolean;
  extract_text?: boolean;
  detect_language?: boolean;
}

export interface PDFUploadResponse {
  success: boolean;
  document?: PDFDocument;
  message: string;
  processing_started: boolean;
  request_id: string;
}

export interface PDFEditRequest {
  page_number: number;
  new_content: string;
  preserve_formatting?: boolean;
}

export interface PDFEditResponse {
  success: boolean;
  edited_page?: EditedPage;
  message: string;
}

export interface VariantGenerateRequest {
  document_id: string;
  number_of_variants?: number;
  continuous_generation?: boolean;
  configuration?: VariantConfiguration;
  stop_condition?: string;
}

export interface VariantGenerateResponse {
  success: boolean;
  variants: PDFVariant[];
  total_generated: number;
  generation_time: number;
  message: string;
  is_stopped: boolean;
  request_id: string;
}

export interface TopicExtractRequest {
  document_id: string;
  min_relevance?: number;
  max_topics?: number;
}

export interface TopicExtractResponse {
  success: boolean;
  topics: TopicItem[];
  main_topic?: string;
  total_topics: number;
  extraction_time: number;
  request_id: string;
}

export interface BrainstormGenerateRequest {
  document_id: string;
  topics?: string[];
  number_of_ideas?: number;
  diversity_level?: number;
}

export interface BrainstormGenerateResponse {
  success: boolean;
  ideas: BrainstormIdea[];
  total_ideas: number;
  generation_time: number;
  message: string;
  request_id: string;
}

export interface VariantStopRequest {
  document_id: string;
  keep_generated?: boolean;
}

export interface VariantStopResponse {
  success: boolean;
  message: string;
  stopped_variants_count: number;
}

export interface SearchRequest {
  query: string;
  document_ids?: string[];
  variant_ids?: string[];
  filters?: Record<string, unknown>;
  limit?: number;
  offset?: number;
}

export interface SearchResult {
  id: string;
  type: "document" | "variant" | "topic";
  title?: string;
  content: string;
  relevance_score: number;
  metadata?: Record<string, unknown>;
}

export interface SearchResponse {
  success: boolean;
  results: SearchResult[];
  total_results: number;
  query: string;
}

export interface BatchProcessingRequest {
  document_ids: string[];
  operation: string;
  configuration?: Record<string, unknown>;
}

export interface BatchProcessingResponse {
  success: boolean;
  processed_count: number;
  failed_count: number;
  results: Array<{
    document_id: string;
    success: boolean;
    error?: string;
  }>;
  total_time: number;
  request_id: string;
}

export interface ExportRequest {
  document_id: string;
  export_format: "pdf" | "docx" | "txt" | "json" | "html";
  include_variants?: boolean;
  include_topics?: boolean;
  options?: Record<string, unknown>;
}

export interface ExportResponse {
  success: boolean;
  file_id: string;
  file_url?: string;
  message: string;
  export_format: string;
  request_id: string;
}

export interface CollaborationInvite {
  document_id: string;
  email: string;
  role: string;
  permissions: string[];
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "unhealthy";
  message: string;
  timestamp: string;
  services: Record<string, {
    status: string;
    message?: string;
  }>;
  version: string;
}

export interface AnalyticsReport {
  period_start: string;
  period_end: string;
  total_documents: number;
  total_variants: number;
  total_topics: number;
  average_processing_time: number;
  metrics: Record<string, unknown>;
}

// API Response Wrapper
export interface APIResponse<T> {
  data?: T;
  error?: {
    message: string;
    code?: string | number;
    details?: Record<string, unknown>;
  };
  status: number;
}

// Pagination
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

// List Query Parameters
export interface ListQueryParams {
  limit?: number;
  offset?: number;
  sort?: string;
  order?: "asc" | "desc";
  [key: string]: unknown;
}






