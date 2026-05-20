/**
 * Sistema de Tipos Estricto para TruthGPT Model Builder
 * ======================================================
 * 
 * Tipos e interfaces estrictas para garantizar type safety en toda la aplicación
 */

// ============================================================================
// TIPOS BASE
// ============================================================================

/**
 * Estados posibles de un modelo
 */
export type ModelStatus = 
  | 'idle'
  | 'validating'
  | 'creating'
  | 'compiling'
  | 'training'
  | 'evaluating'
  | 'predicting'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'paused'

/**
 * Tipos de modelo soportados
 */
export type ModelType = 
  | 'classification'
  | 'regression'
  | 'nlp'
  | 'vision'
  | 'time-series'
  | 'generative'
  | 'custom'

/**
 * Arquitecturas de modelo soportadas
 */
export type ModelArchitecture = 
  | 'dense'
  | 'cnn'
  | 'rnn'
  | 'lstm'
  | 'gru'
  | 'transformer'
  | 'hybrid'
  | 'auto'

/**
 * Optimizadores soportados
 */
export type OptimizerType = 
  | 'adam'
  | 'sgd'
  | 'rmsprop'
  | 'adagrad'
  | 'adamw'
  | 'nadam'

/**
 * Funciones de pérdida soportadas
 */
export type LossFunction = 
  | 'sparse_categorical_crossentropy'
  | 'categorical_crossentropy'
  | 'binary_crossentropy'
  | 'mean_squared_error'
  | 'mean_absolute_error'
  | 'huber'
  | 'cosine_similarity'

/**
 * Métricas soportadas
 */
export type MetricType = 
  | 'accuracy'
  | 'precision'
  | 'recall'
  | 'f1_score'
  | 'auc'
  | 'mae'
  | 'mse'
  | 'rmse'
  | 'r2_score'

// ============================================================================
// CONFIGURACIÓN DE CAPAS
// ============================================================================

/**
 * Tipos de capas soportadas
 */
export type LayerType = 
  | 'Dense'
  | 'Conv2D'
  | 'Conv1D'
  | 'LSTM'
  | 'GRU'
  | 'Dropout'
  | 'BatchNormalization'
  | 'MaxPooling2D'
  | 'AveragePooling2D'
  | 'Flatten'
  | 'Reshape'
  | 'Embedding'
  | 'Attention'
  | 'GlobalAveragePooling2D'
  | 'GlobalMaxPooling2D'

/**
 * Configuración de una capa
 */
export interface LayerConfig {
  readonly type: LayerType
  readonly params: Readonly<Record<string, unknown>>
  readonly name?: string
  readonly activation?: string
}

/**
 * Configuración de capa Dense
 */
export interface DenseLayerConfig extends LayerConfig {
  readonly type: 'Dense'
  readonly params: Readonly<{
    units: number
    activation?: string
    use_bias?: boolean
    kernel_initializer?: string
    bias_initializer?: string
  }>
}

/**
 * Configuración de capa Conv2D
 */
export interface Conv2DLayerConfig extends LayerConfig {
  readonly type: 'Conv2D'
  readonly params: Readonly<{
    filters: number
    kernel_size: [number, number] | number
    strides?: [number, number] | number
    padding?: 'valid' | 'same'
    activation?: string
  }>
}

/**
 * Configuración de capa LSTM
 */
export interface LSTMLayerConfig extends LayerConfig {
  readonly type: 'LSTM'
  readonly params: Readonly<{
    units: number
    return_sequences?: boolean
    dropout?: number
    recurrent_dropout?: number
  }>
}

// ============================================================================
// ESPECIFICACIÓN DE MODELO
// ============================================================================

/**
 * Especificación completa de un modelo
 */
export interface ModelSpec {
  readonly type: ModelType
  readonly architecture: ModelArchitecture
  readonly layers: readonly LayerConfig[]
  readonly inputShape?: readonly number[]
  readonly outputShape?: readonly number[]
  readonly optimizer: OptimizerConfig
  readonly loss: LossConfig
  readonly metrics: readonly MetricType[]
  readonly training?: TrainingConfig
  readonly compilation?: CompilationConfig
}

/**
 * Configuración de optimizador
 */
export interface OptimizerConfig {
  readonly type: OptimizerType
  readonly params?: Readonly<Record<string, unknown>>
  readonly learningRate?: number
}

/**
 * Configuración de función de pérdida
 */
export interface LossConfig {
  readonly type: LossFunction
  readonly params?: Readonly<Record<string, unknown>>
}

/**
 * Configuración de entrenamiento
 */
export interface TrainingConfig {
  readonly epochs: number
  readonly batchSize: number
  readonly validationSplit?: number
  readonly shuffle?: boolean
  readonly verbose?: number
  readonly callbacks?: readonly CallbackConfig[]
  readonly earlyStopping?: EarlyStoppingConfig
}

/**
 * Configuración de compilación
 */
export interface CompilationConfig {
  readonly runEagerly?: boolean
  readonly stepsPerExecution?: number
}

/**
 * Configuración de callback
 */
export interface CallbackConfig {
  readonly type: string
  readonly params?: Readonly<Record<string, unknown>>
}

/**
 * Configuración de early stopping
 */
export interface EarlyStoppingConfig {
  readonly monitor: string
  readonly patience: number
  readonly minDelta?: number
  readonly restoreBestWeights?: boolean
}

// ============================================================================
// ESTADO Y PROGRESO
// ============================================================================

/**
 * Estado detallado de un modelo
 */
export interface ModelState {
  readonly id: string
  readonly name: string
  readonly status: ModelStatus
  readonly progress: number // 0-100
  readonly currentStep?: string
  readonly error?: ModelError
  readonly createdAt: Date
  readonly updatedAt: Date
  readonly completedAt?: Date
  readonly spec?: ModelSpec
  readonly metadata?: Readonly<Record<string, unknown>>
}

/**
 * Error de modelo tipado
 */
export interface ModelError {
  readonly code: string
  readonly message: string
  readonly details?: Readonly<Record<string, unknown>>
  readonly timestamp: Date
  readonly retryable: boolean
}

/**
 * Progreso de creación de modelo
 */
export interface ModelProgress {
  readonly modelId: string
  readonly step: ModelCreationStep
  readonly progress: number // 0-100
  readonly message: string
  readonly estimatedTimeRemaining?: number // segundos
  readonly startedAt: Date
  readonly updatedAt: Date
}

/**
 * Pasos de creación de modelo
 */
export type ModelCreationStep = 
  | 'validation'
  | 'layer_parsing'
  | 'model_creation'
  | 'compilation'
  | 'initialization'
  | 'ready'

// ============================================================================
// RESULTADOS Y MÉTRICAS
// ============================================================================

/**
 * Resultados de entrenamiento
 */
export interface TrainingResults {
  readonly modelId: string
  readonly history: TrainingHistory
  readonly finalMetrics: Readonly<Record<string, number>>
  readonly bestEpoch: number
  readonly trainingTime: number // segundos
  readonly completedAt: Date
}

/**
 * Historial de entrenamiento
 */
export interface TrainingHistory {
  readonly loss: readonly number[]
  readonly val_loss?: readonly number[]
  readonly accuracy?: readonly number[]
  readonly val_accuracy?: readonly number[]
  readonly [metric: string]: readonly number[] | undefined
}

/**
 * Resultados de evaluación
 */
export interface EvaluationResults {
  readonly modelId: string
  readonly metrics: Readonly<Record<string, number>>
  readonly confusionMatrix?: readonly (readonly number[])[]
  readonly evaluatedAt: Date
}

/**
 * Resultados de predicción
 */
export interface PredictionResults {
  readonly modelId: string
  readonly predictions: readonly (readonly number[])[]
  readonly probabilities?: readonly (readonly number[])[]
  readonly predictedAt: Date
}

// ============================================================================
// VALIDACIÓN
// ============================================================================

/**
 * Resultado de validación
 */
export interface ValidationResult {
  readonly valid: boolean
  readonly errors: readonly ValidationError[]
  readonly warnings: readonly ValidationWarning[]
  readonly score: number // 0-100
}

/**
 * Error de validación
 */
export interface ValidationError {
  readonly field: string
  readonly message: string
  readonly code: string
  readonly severity: 'error' | 'critical'
}

/**
 * Advertencia de validación
 */
export interface ValidationWarning {
  readonly field: string
  readonly message: string
  readonly code: string
  readonly suggestion?: string
}

// ============================================================================
// ANALYTICS Y ESTADÍSTICAS
// ============================================================================

/**
 * Estadísticas de analytics
 */
export interface AnalyticsStats {
  readonly totalCreated: number
  readonly totalCompleted: number
  readonly totalFailed: number
  readonly averageCreationTime: number // segundos
  readonly successRate: number // 0-1
  readonly errorRate: number // 0-1
  readonly mostCommonErrors: ReadonlyMap<string, number>
  readonly timeSeries: readonly AnalyticsDataPoint[]
}

/**
 * Punto de datos de analytics
 */
export interface AnalyticsDataPoint {
  readonly timestamp: Date
  readonly created: number
  readonly completed: number
  readonly failed: number
}

// ============================================================================
// HISTORIAL Y PLANTILLAS
// ============================================================================

/**
 * Entrada de historial
 */
export interface HistoryEntry {
  readonly id: string
  readonly modelId?: string
  readonly name: string
  readonly description: string
  readonly spec?: ModelSpec
  readonly status: ModelStatus
  readonly tags: readonly string[]
  readonly notes?: string
  readonly createdAt: Date
  readonly updatedAt: Date
}

/**
 * Plantilla de modelo
 */
export interface ModelTemplate {
  readonly id: string
  readonly name: string
  readonly description: string
  readonly category: string
  readonly spec: ModelSpec
  readonly tags: readonly string[]
  readonly isDefault: boolean
  readonly createdAt: Date
  readonly usageCount: number
}

// ============================================================================
// COMPARACIÓN
// ============================================================================

/**
 * Comparación de modelos
 */
export interface ModelComparison {
  readonly id: string
  readonly modelIds: readonly string[]
  readonly metrics: readonly ComparisonMetric[]
  readonly createdAt: Date
}

/**
 * Métrica de comparación
 */
export interface ComparisonMetric {
  readonly name: string
  readonly values: ReadonlyMap<string, number>
  readonly bestModelId: string
  readonly worstModelId: string
}

// ============================================================================
// NOTIFICACIONES
// ============================================================================

/**
 * Tipo de notificación
 */
export type NotificationType = 
  | 'success'
  | 'error'
  | 'warning'
  | 'info'
  | 'loading'

/**
 * Notificación
 */
export interface Notification {
  readonly id: string
  readonly type: NotificationType
  readonly title: string
  readonly message?: string
  readonly duration?: number // milisegundos
  readonly timestamp: Date
  readonly actions?: readonly NotificationAction[]
}

/**
 * Acción de notificación
 */
export interface NotificationAction {
  readonly label: string
  readonly action: () => void | Promise<void>
}

// ============================================================================
// CACHE Y PERFORMANCE
// ============================================================================

/**
 * Estadísticas de caché
 */
export interface CacheStats {
  readonly size: number
  readonly maxSize: number
  readonly hitRate: number // 0-1
  readonly missRate: number // 0-1
  readonly evictions: number
  readonly lastAccess?: Date
}

/**
 * Métricas de performance
 */
export interface PerformanceMetrics {
  readonly apiCallCount: number
  readonly averageResponseTime: number // milisegundos
  readonly cacheHitRate: number // 0-1
  readonly errorRate: number // 0-1
  readonly throughput: number // requests/segundo
}

// ============================================================================
// COLA DE PROCESAMIENTO
// ============================================================================

/**
 * Estado de cola
 */
export interface QueueStats {
  readonly total: number
  readonly pending: number
  readonly processing: number
  readonly completed: number
  readonly failed: number
  readonly averageWaitTime: number // segundos
  readonly averageProcessingTime: number // segundos
}

/**
 * Item de cola
 */
export interface QueueItem {
  readonly id: string
  readonly priority: number
  readonly status: 'pending' | 'processing' | 'completed' | 'failed'
  readonly createdAt: Date
  readonly startedAt?: Date
  readonly completedAt?: Date
  readonly data: Readonly<Record<string, unknown>>
}

// ============================================================================
// UTILIDADES DE TIPO
// ============================================================================

/**
 * Hace todos los campos opcionales excepto los especificados
 */
export type PartialExcept<T, K extends keyof T> = Partial<T> & Pick<T, K>

/**
 * Hace todos los campos requeridos
 */
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>

/**
 * Extrae el tipo de retorno de una función
 */
export type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never

/**
 * Hace un tipo readonly profundo
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P]
}







