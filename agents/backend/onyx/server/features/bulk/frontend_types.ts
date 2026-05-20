/**
 * BUL API - TypeScript Types
 * ============================
 * 
 * Tipos TypeScript para el consumo de la API desde el frontend
 */

// ============================================================================
// Request Types
// ============================================================================

export interface DocumentRequest {
  /** Consulta de negocio para generación de documento (10-5000 caracteres) */
  query: string;
  /** Área de negocio específica (opcional) */
  business_area?: string;
  /** Tipo de documento a generar (opcional) */
  document_type?: string;
  /** Prioridad de procesamiento (1-5, por defecto 1) */
  priority?: number;
  /** Metadatos adicionales (opcional) */
  metadata?: Record<string, any>;
  /** ID de usuario (opcional) */
  user_id?: string;
  /** ID de sesión (opcional) */
  session_id?: string;
}

// ============================================================================
// Response Types
// ============================================================================

export interface DocumentResponse {
  /** ID de la tarea generada */
  task_id: string;
  /** Estado de la tarea */
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  /** Mensaje descriptivo */
  message: string;
  /** Tiempo estimado de procesamiento en segundos */
  estimated_time?: number;
  /** Posición en la cola */
  queue_position?: number;
  /** Fecha de creación (ISO 8601) */
  created_at: string;
}

export interface TaskStatus {
  /** ID de la tarea */
  task_id: string;
  /** Estado actual */
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  /** Progreso (0-100) */
  progress: number;
  /** Resultado del procesamiento (si está completado) */
  result?: {
    content: string;
    format: string;
    word_count: number;
    generated_at: string;
  };
  /** Error (si falló) */
  error?: string;
  /** Fecha de creación (ISO 8601) */
  created_at: string;
  /** Fecha de última actualización (ISO 8601) */
  updated_at: string;
  /** Tiempo de procesamiento en segundos */
  processing_time?: number;
}

export interface HealthResponse {
  /** Estado del sistema */
  status: 'healthy' | 'unhealthy';
  /** Timestamp (ISO 8601) */
  timestamp: string;
  /** Tiempo de actividad */
  uptime: string;
  /** Número de tareas activas */
  active_tasks: number;
  /** Total de solicitudes */
  total_requests: number;
  /** Versión de la API */
  version: string;
}

export interface StatsResponse {
  /** Total de solicitudes */
  total_requests: number;
  /** Tareas activas */
  active_tasks: number;
  /** Tareas completadas */
  completed_tasks: number;
  /** Tasa de éxito (0-1) */
  success_rate: number;
  /** Tiempo promedio de procesamiento en segundos */
  average_processing_time: number;
  /** Tiempo de actividad */
  uptime: string;
}

export interface TaskListItem {
  /** ID de la tarea */
  task_id: string;
  /** Estado */
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  /** Progreso (0-100) */
  progress: number;
  /** Fecha de creación (ISO 8601) */
  created_at: string;
  /** Fecha de última actualización (ISO 8601) */
  updated_at: string;
  /** ID de usuario */
  user_id?: string;
  /** Vista previa de la consulta */
  query_preview: string;
}

export interface TaskListResponse {
  /** Lista de tareas */
  tasks: TaskListItem[];
  /** Total de tareas */
  total: number;
  /** Límite de resultados */
  limit: number;
  /** Offset de paginación */
  offset: number;
  /** Indica si hay más resultados */
  has_more: boolean;
}

export interface DocumentListItem {
  /** ID de la tarea */
  task_id: string;
  /** Fecha de creación (ISO 8601) */
  created_at: string;
  /** Vista previa de la consulta */
  query_preview: string;
  /** Área de negocio */
  business_area?: string;
  /** Tipo de documento */
  document_type?: string;
}

export interface DocumentListResponse {
  /** Lista de documentos */
  documents: DocumentListItem[];
  /** Total de documentos */
  total: number;
  /** Límite de resultados */
  limit: number;
  /** Offset de paginación */
  offset: number;
  /** Indica si hay más resultados */
  has_more: boolean;
}

export interface TaskDocumentResponse {
  /** ID de la tarea */
  task_id: string;
  /** Contenido del documento */
  document: {
    content: string;
    format: string;
    word_count: number;
    generated_at: string;
  };
  /** Metadatos de la solicitud original */
  metadata: DocumentRequest;
  /** Fecha de creación (ISO 8601) */
  created_at: string;
  /** Fecha de finalización (ISO 8601) */
  completed_at: string;
}

export interface ErrorResponse {
  /** Detalle del error */
  detail: string;
}

// ============================================================================
// Utility Types
// ============================================================================

export type TaskStatusType = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';

export interface ApiConfig {
  /** URL base de la API */
  baseUrl: string;
  /** Timeout en milisegundos */
  timeout?: number;
  /** Headers adicionales */
  headers?: Record<string, string>;
}

export interface WebSocketMessage {
  /** Tipo de mensaje */
  type: 'task_update' | 'initial_state' | 'connected' | 'error' | 'pong';
  /** ID de la tarea (si aplica) */
  task_id?: string;
  /** Datos del mensaje */
  data?: any;
  /** Mensaje de texto (para errores) */
  message?: string;
  /** Timestamp del mensaje */
  timestamp: string;
}

