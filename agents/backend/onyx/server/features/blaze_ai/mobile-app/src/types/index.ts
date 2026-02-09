// Core system interfaces
export interface BaseModule {
  id: string;
  name: string;
  version: string;
  isActive: boolean;
  healthStatus: HealthStatus;
  dependencies: string[];
  metrics: ModuleMetrics;
}

export interface HealthStatus {
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  lastCheck: Date;
  message?: string;
  details?: Record<string, unknown>;
}

export interface ModuleMetrics {
  cpuUsage: number;
  memoryUsage: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  lastUpdated: Date;
}

// Cache module interfaces
export interface CacheModule extends BaseModule {
  cacheType: 'lru' | 'lfu' | 'fifo' | 'ttl' | 'size-based' | 'hybrid';
  compressionType: 'lz4' | 'zlib' | 'snappy' | 'pickle';
  maxSize: number;
  currentSize: number;
  hitRate: number;
  missRate: number;
  evictionCount: number;
}

export interface CacheEntry<T = unknown> {
  key: string;
  value: T;
  tags: string[];
  createdAt: Date;
  expiresAt?: Date;
  accessCount: number;
  lastAccessed: Date;
  size: number;
}

// Monitoring module interfaces
export interface MonitoringModule extends BaseModule {
  metricsCollectors: MetricsCollector[];
  alertRules: AlertRule[];
  thresholds: ThresholdConfig[];
  isPersistent: boolean;
}

export interface MetricsCollector {
  id: string;
  name: string;
  type: 'system' | 'custom' | 'business';
  interval: number;
  isActive: boolean;
  lastCollection: Date;
}

export interface AlertRule {
  id: string;
  name: string;
  condition: AlertCondition;
  severity: 'low' | 'medium' | 'high' | 'critical';
  isActive: boolean;
  lastTriggered?: Date;
}

export interface AlertCondition {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'ne';
  value: number;
  duration: number;
}

export interface ThresholdConfig {
  metric: string;
  warning: number;
  critical: number;
  action: string;
}

// Optimization module interfaces
export interface OptimizationModule extends BaseModule {
  algorithms: OptimizationAlgorithm[];
  currentTask?: OptimizationTask;
  taskQueue: OptimizationTask[];
  convergenceHistory: ConvergencePoint[];
}

export interface OptimizationAlgorithm {
  id: string;
  name: string;
  type: 'genetic' | 'simulated-annealing' | 'particle-swarm' | 'gradient-descent';
  parameters: Record<string, unknown>;
  isActive: boolean;
}

export interface OptimizationTask {
  id: string;
  name: string;
  algorithm: string;
  parameters: Record<string, unknown>;
  constraints: Constraint[];
  priority: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result?: OptimizationResult;
}

export interface Constraint {
  name: string;
  type: 'equality' | 'inequality' | 'bound';
  expression: string;
  penalty: number;
}

export interface OptimizationResult {
  optimalValue: number;
  optimalParameters: Record<string, unknown>;
  iterations: number;
  convergenceTime: number;
  constraints: ConstraintEvaluation[];
}

export interface ConstraintEvaluation {
  constraint: string;
  isSatisfied: boolean;
  violation: number;
}

export interface ConvergencePoint {
  iteration: number;
  value: number;
  timestamp: Date;
}

// Storage module interfaces
export interface StorageModule extends BaseModule {
  compressionTypes: string[];
  deduplicationEnabled: boolean;
  encryptionEnabled: boolean;
  hybridStorageEnabled: boolean;
  storageStats: StorageStats;
}

export interface StorageStats {
  totalSpace: number;
  usedSpace: number;
  availableSpace: number;
  compressionRatio: number;
  deduplicationRatio: number;
  encryptionOverhead: number;
}

// Execution module interfaces
export interface ExecutionModule extends BaseModule {
  workers: Worker[];
  taskQueue: ExecutionTask[];
  loadBalancingStrategy: 'round-robin' | 'least-connections' | 'adaptive';
  autoScalingEnabled: boolean;
  maxWorkers: number;
  currentLoad: number;
}

export interface Worker {
  id: string;
  status: 'idle' | 'busy' | 'offline';
  currentTask?: string;
  performance: WorkerPerformance;
  lastHeartbeat: Date;
}

export interface WorkerPerformance {
  cpuUsage: number;
  memoryUsage: number;
  throughput: number;
  errorRate: number;
}

export interface ExecutionTask {
  id: string;
  type: string;
  priority: number;
  status: 'queued' | 'running' | 'completed' | 'failed';
  assignedWorker?: string;
  retryCount: number;
  maxRetries: number;
  timeout: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
}

// Engines module interfaces
export interface EnginesModule extends BaseModule {
  engines: Engine[];
  hybridEngineEnabled: boolean;
  engineHealth: Record<string, HealthStatus>;
}

export interface Engine {
  id: string;
  name: string;
  type: 'quantum' | 'neural-turbo' | 'marareal' | 'hybrid';
  status: 'active' | 'inactive' | 'error';
  performance: EnginePerformance;
  capabilities: string[];
}

export interface EnginePerformance {
  speed: number;
  accuracy: number;
  efficiency: number;
  reliability: number;
}

// ML module interfaces
export interface MLModule extends BaseModule {
  models: MLModel[];
  trainingJobs: TrainingJob[];
  autoMLEnabled: boolean;
  experimentTracking: Experiment[];
}

export interface MLModel {
  id: string;
  name: string;
  type: string;
  version: string;
  status: 'training' | 'ready' | 'deployed' | 'archived';
  performance: ModelPerformance;
  metadata: ModelMetadata;
}

export interface ModelPerformance {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  inferenceTime: number;
}

export interface ModelMetadata {
  algorithm: string;
  hyperparameters: Record<string, unknown>;
  trainingData: string;
  validationData: string;
  createdAt: Date;
  lastUpdated: Date;
}

export interface TrainingJob {
  id: string;
  modelId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime?: Date;
  endTime?: Date;
  metrics: TrainingMetrics;
}

export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  validationLoss: number;
  validationAccuracy: number;
  epoch: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed';
  metrics: Record<string, unknown>;
  artifacts: string[];
  createdAt: Date;
  completedAt?: Date;
}

// Data analysis module interfaces
export interface DataAnalysisModule extends BaseModule {
  dataSources: DataSource[];
  analysisJobs: AnalysisJob[];
  dataQuality: DataQualityMetrics;
  patterns: Pattern[];
}

export interface DataSource {
  id: string;
  name: string;
  type: 'csv' | 'json' | 'excel' | 'database' | 'api';
  location: string;
  format: string;
  size: number;
  lastUpdated: Date;
}

export interface AnalysisJob {
  id: string;
  name: string;
  dataSourceId: string;
  type: 'descriptive' | 'exploratory' | 'clustering' | 'classification';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result?: AnalysisResult;
}

export interface AnalysisResult {
  summary: Record<string, unknown>;
  visualizations: string[];
  insights: string[];
  recommendations: string[];
}

export interface DataQualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  validity: number;
}

export interface Pattern {
  id: string;
  name: string;
  type: string;
  confidence: number;
  description: string;
  examples: string[];
}

// AI Intelligence module interfaces
export interface AIIntelligenceModule extends BaseModule {
  nlpCapabilities: NLPCapability[];
  computerVision: ComputerVisionCapability[];
  reasoning: ReasoningCapability[];
  multimodal: MultimodalCapability[];
}

export interface NLPCapability {
  id: string;
  name: string;
  type: 'sentiment-analysis' | 'text-classification' | 'named-entity-recognition' | 'summarization';
  accuracy: number;
  languages: string[];
  isActive: boolean;
}

export interface ComputerVisionCapability {
  id: string;
  name: string;
  type: 'object-detection' | 'image-classification' | 'face-recognition' | 'ocr';
  accuracy: number;
  supportedFormats: string[];
  isActive: boolean;
}

export interface ReasoningCapability {
  id: string;
  name: string;
  type: 'logical' | 'symbolic' | 'quantum';
  complexity: 'basic' | 'intermediate' | 'advanced';
  isActive: boolean;
}

export interface MultimodalCapability {
  id: string;
  name;
  name: string;
  inputTypes: string[];
  outputTypes: string[];
  fusionStrategy: string;
  isActive: boolean;
}

// API REST module interfaces
export interface APIRESTModule extends BaseModule {
  endpoints: APIEndpoint[];
  authentication: AuthenticationConfig;
  rateLimiting: RateLimitConfig;
  cors: CORSConfig;
  documentation: DocumentationConfig;
}

export interface APIEndpoint {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  description: string;
  parameters: Parameter[];
  responses: Response[];
  rateLimit?: number;
  requiresAuth: boolean;
}

export interface Parameter {
  name: string;
  type: string;
  required: boolean;
  description: string;
  defaultValue?: unknown;
}

export interface Response {
  code: number;
  description: string;
  schema: Record<string, unknown>;
}

export interface AuthenticationConfig {
  enabled: boolean;
  methods: ('api-key' | 'jwt' | 'oauth2')[];
  apiKeyHeader: string;
  jwtSecret?: string;
  oauthProviders: OAuthProvider[];
}

export interface OAuthProvider {
  name: string;
  clientId: string;
  clientSecret: string;
  authorizationUrl: string;
  tokenUrl: string;
}

export interface RateLimitConfig {
  enabled: boolean;
  requestsPerMinute: number;
  burstSize: number;
  strategy: 'fixed-window' | 'sliding-window' | 'token-bucket';
}

export interface CORSConfig {
  enabled: boolean;
  allowedOrigins: string[];
  allowedMethods: string[];
  allowedHeaders: string[];
  allowCredentials: boolean;
}

export interface DocumentationConfig {
  swaggerEnabled: boolean;
  redocEnabled: boolean;
  customDocs?: string;
}

// Security module interfaces
export interface SecurityModule extends BaseModule {
  encryption: EncryptionConfig;
  authentication: SecurityAuthenticationConfig;
  authorization: AuthorizationConfig;
  audit: AuditConfig;
  threatDetection: ThreatDetectionConfig;
}

export interface EncryptionConfig {
  algorithm: string;
  keySize: number;
  keyRotationEnabled: boolean;
  rotationInterval: number;
}

export interface SecurityAuthenticationConfig {
  methods: string[];
  mfaEnabled: boolean;
  sessionTimeout: number;
  maxLoginAttempts: number;
}

export interface AuthorizationConfig {
  rbacEnabled: boolean;
  permissions: Permission[];
  roles: Role[];
}

export interface Permission {
  id: string;
  name: string;
  resource: string;
  action: string;
}

export interface Role {
  id: string;
  name: string;
  permissions: string[];
  description: string;
}

export interface AuditConfig {
  enabled: boolean;
  logLevel: 'info' | 'warning' | 'error';
  retentionPeriod: number;
  sensitiveActions: string[];
}

export interface ThreatDetectionConfig {
  enabled: boolean;
  rules: ThreatRule[];
  autoResponse: boolean;
  notificationChannels: string[];
}

export interface ThreatRule {
  id: string;
  name: string;
  pattern: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  action: string;
}

// Distributed processing module interfaces
export interface DistributedProcessingModule extends BaseModule {
  nodes: Node[];
  loadBalancing: LoadBalancingConfig;
  faultTolerance: FaultToleranceConfig;
  autoScaling: AutoScalingConfig;
  taskScheduling: TaskSchedulingConfig;
}

export interface Node {
  id: string;
  hostname: string;
  ip: string;
  status: 'online' | 'offline' | 'maintenance';
  resources: NodeResources;
  lastHeartbeat: Date;
}

export interface NodeResources {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
}

export interface LoadBalancingConfig {
  strategy: 'round-robin' | 'least-connections' | 'adaptive' | 'weighted';
  healthCheckEnabled: boolean;
  healthCheckInterval: number;
  failoverEnabled: boolean;
}

export interface FaultToleranceConfig {
  circuitBreakerEnabled: boolean;
  replicationFactor: number;
  checkpointingEnabled: boolean;
  recoveryStrategy: string;
}

export interface AutoScalingConfig {
  enabled: boolean;
  minNodes: number;
  maxNodes: number;
  scaleUpThreshold: number;
  scaleDownThreshold: number;
  cooldownPeriod: number;
}

export interface TaskSchedulingConfig {
  strategy: 'fifo' | 'priority' | 'fair' | 'deadline';
  preemptionEnabled: boolean;
  resourceConstraints: boolean;
  retryPolicy: RetryPolicy;
}

export interface RetryPolicy {
  maxRetries: number;
  backoffStrategy: 'fixed' | 'exponential' | 'linear';
  initialDelay: number;
  maxDelay: number;
}

// System configuration interfaces
export interface SystemConfig {
  app: AppConfig;
  database: DatabaseConfig;
  cache: CacheConfig;
  security: SecurityConfig;
  monitoring: MonitoringConfig;
  api: APIConfig;
}

export interface AppConfig {
  name: string;
  version: string;
  environment: 'development' | 'staging' | 'production';
  debug: boolean;
  logLevel: string;
}

export interface DatabaseConfig {
  host: string;
  port: number;
  name: string;
  username: string;
  password: string;
  poolSize: number;
  ssl: boolean;
}

export interface CacheConfig {
  type: string;
  host: string;
  port: number;
  password?: string;
  maxMemory: number;
  ttl: number;
}

export interface SecurityConfig {
  jwtSecret: string;
  bcryptRounds: number;
  sessionTimeout: number;
  corsOrigins: string[];
}

export interface MonitoringConfig {
  enabled: boolean;
  interval: number;
  metricsRetention: number;
  alertChannels: string[];
}

export interface APIConfig {
  host: string;
  port: number;
  corsEnabled: boolean;
  rateLimitEnabled: boolean;
  docsEnabled: boolean;
}

// Error handling interfaces
export interface BlazeAIError extends Error {
  code: string;
  statusCode: number;
  details?: Record<string, unknown>;
  timestamp: Date;
  requestId?: string;
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
    timestamp: string;
    requestId?: string;
  };
}

// Response interfaces
export interface APIResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  requestId?: string;
}

export interface PaginatedResponse<T = unknown> extends APIResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

// Navigation interfaces
export interface NavigationProps {
  navigation: any;
  route: any;
}

export interface TabNavigationProps {
  navigation: any;
  route: any;
}

// Theme interfaces
export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    warning: string;
    success: string;
    info: string;
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
    xxl: number;
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
  typography: {
    h1: TextStyle;
    h2: TextStyle;
    h3: TextStyle;
    h4: TextStyle;
    h5: TextStyle;
    h6: TextStyle;
    body: TextStyle;
    caption: TextStyle;
    button: TextStyle;
  };
}

export interface TextStyle {
  fontSize: number;
  fontWeight: 'normal' | 'bold' | '100' | '200' | '300' | '400' | '500' | '600' | '700' | '800' | '900';
  lineHeight: number;
  letterSpacing?: number;
}

// Component props interfaces
export interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: string;
}

export interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  onPress?: () => void;
  variant?: 'default' | 'elevated' | 'outlined';
}

export interface InputProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChangeText: (text: string) => void;
  error?: string;
  disabled?: boolean;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad';
}

export interface ModalProps {
  visible: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  showCloseButton?: boolean;
}
export interface BaseModule {
  id: string;
  name: string;
  version: string;
  isActive: boolean;
  healthStatus: HealthStatus;
  dependencies: string[];
  metrics: ModuleMetrics;
}

export interface HealthStatus {
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  lastCheck: Date;
  message?: string;
  details?: Record<string, unknown>;
}

export interface ModuleMetrics {
  cpuUsage: number;
  memoryUsage: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  lastUpdated: Date;
}

// Cache module interfaces
export interface CacheModule extends BaseModule {
  cacheType: 'lru' | 'lfu' | 'fifo' | 'ttl' | 'size-based' | 'hybrid';
  compressionType: 'lz4' | 'zlib' | 'snappy' | 'pickle';
  maxSize: number;
  currentSize: number;
  hitRate: number;
  missRate: number;
  evictionCount: number;
}

export interface CacheEntry<T = unknown> {
  key: string;
  value: T;
  tags: string[];
  createdAt: Date;
  expiresAt?: Date;
  accessCount: number;
  lastAccessed: Date;
  size: number;
}

// Monitoring module interfaces
export interface MonitoringModule extends BaseModule {
  metricsCollectors: MetricsCollector[];
  alertRules: AlertRule[];
  thresholds: ThresholdConfig[];
  isPersistent: boolean;
}

export interface MetricsCollector {
  id: string;
  name: string;
  type: 'system' | 'custom' | 'business';
  interval: number;
  isActive: boolean;
  lastCollection: Date;
}

export interface AlertRule {
  id: string;
  name: string;
  condition: AlertCondition;
  severity: 'low' | 'medium' | 'high' | 'critical';
  isActive: boolean;
  lastTriggered?: Date;
}

export interface AlertCondition {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'ne';
  value: number;
  duration: number;
}

export interface ThresholdConfig {
  metric: string;
  warning: number;
  critical: number;
  action: string;
}

// Optimization module interfaces
export interface OptimizationModule extends BaseModule {
  algorithms: OptimizationAlgorithm[];
  currentTask?: OptimizationTask;
  taskQueue: OptimizationTask[];
  convergenceHistory: ConvergencePoint[];
}

export interface OptimizationAlgorithm {
  id: string;
  name: string;
  type: 'genetic' | 'simulated-annealing' | 'particle-swarm' | 'gradient-descent';
  parameters: Record<string, unknown>;
  isActive: boolean;
}

export interface OptimizationTask {
  id: string;
  name: string;
  algorithm: string;
  parameters: Record<string, unknown>;
  constraints: Constraint[];
  priority: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result?: OptimizationResult;
}

export interface Constraint {
  name: string;
  type: 'equality' | 'inequality' | 'bound';
  expression: string;
  penalty: number;
}

export interface OptimizationResult {
  optimalValue: number;
  optimalParameters: Record<string, unknown>;
  iterations: number;
  convergenceTime: number;
  constraints: ConstraintEvaluation[];
}

export interface ConstraintEvaluation {
  constraint: string;
  isSatisfied: boolean;
  violation: number;
}

export interface ConvergencePoint {
  iteration: number;
  value: number;
  timestamp: Date;
}

// Storage module interfaces
export interface StorageModule extends BaseModule {
  compressionTypes: string[];
  deduplicationEnabled: boolean;
  encryptionEnabled: boolean;
  hybridStorageEnabled: boolean;
  storageStats: StorageStats;
}

export interface StorageStats {
  totalSpace: number;
  usedSpace: number;
  availableSpace: number;
  compressionRatio: number;
  deduplicationRatio: number;
  encryptionOverhead: number;
}

// Execution module interfaces
export interface ExecutionModule extends BaseModule {
  workers: Worker[];
  taskQueue: ExecutionTask[];
  loadBalancingStrategy: 'round-robin' | 'least-connections' | 'adaptive';
  autoScalingEnabled: boolean;
  maxWorkers: number;
  currentLoad: number;
}

export interface Worker {
  id: string;
  status: 'idle' | 'busy' | 'offline';
  currentTask?: string;
  performance: WorkerPerformance;
  lastHeartbeat: Date;
}

export interface WorkerPerformance {
  cpuUsage: number;
  memoryUsage: number;
  throughput: number;
  errorRate: number;
}

export interface ExecutionTask {
  id: string;
  type: string;
  priority: number;
  status: 'queued' | 'running' | 'completed' | 'failed';
  assignedWorker?: string;
  retryCount: number;
  maxRetries: number;
  timeout: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
}

// Engines module interfaces
export interface EnginesModule extends BaseModule {
  engines: Engine[];
  hybridEngineEnabled: boolean;
  engineHealth: Record<string, HealthStatus>;
}

export interface Engine {
  id: string;
  name: string;
  type: 'quantum' | 'neural-turbo' | 'marareal' | 'hybrid';
  status: 'active' | 'inactive' | 'error';
  performance: EnginePerformance;
  capabilities: string[];
}

export interface EnginePerformance {
  speed: number;
  accuracy: number;
  efficiency: number;
  reliability: number;
}

// ML module interfaces
export interface MLModule extends BaseModule {
  models: MLModel[];
  trainingJobs: TrainingJob[];
  autoMLEnabled: boolean;
  experimentTracking: Experiment[];
}

export interface MLModel {
  id: string;
  name: string;
  type: string;
  version: string;
  status: 'training' | 'ready' | 'deployed' | 'archived';
  performance: ModelPerformance;
  metadata: ModelMetadata;
}

export interface ModelPerformance {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  inferenceTime: number;
}

export interface ModelMetadata {
  algorithm: string;
  hyperparameters: Record<string, unknown>;
  trainingData: string;
  validationData: string;
  createdAt: Date;
  lastUpdated: Date;
}

export interface TrainingJob {
  id: string;
  modelId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime?: Date;
  endTime?: Date;
  metrics: TrainingMetrics;
}

export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  validationLoss: number;
  validationAccuracy: number;
  epoch: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed';
  metrics: Record<string, unknown>;
  artifacts: string[];
  createdAt: Date;
  completedAt?: Date;
}

// Data analysis module interfaces
export interface DataAnalysisModule extends BaseModule {
  dataSources: DataSource[];
  analysisJobs: AnalysisJob[];
  dataQuality: DataQualityMetrics;
  patterns: Pattern[];
}

export interface DataSource {
  id: string;
  name: string;
  type: 'csv' | 'json' | 'excel' | 'database' | 'api';
  location: string;
  format: string;
  size: number;
  lastUpdated: Date;
}

export interface AnalysisJob {
  id: string;
  name: string;
  dataSourceId: string;
  type: 'descriptive' | 'exploratory' | 'clustering' | 'classification';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result?: AnalysisResult;
}

export interface AnalysisResult {
  summary: Record<string, unknown>;
  visualizations: string[];
  insights: string[];
  recommendations: string[];
}

export interface DataQualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  validity: number;
}

export interface Pattern {
  id: string;
  name: string;
  type: string;
  confidence: number;
  description: string;
  examples: string[];
}

// AI Intelligence module interfaces
export interface AIIntelligenceModule extends BaseModule {
  nlpCapabilities: NLPCapability[];
  computerVision: ComputerVisionCapability[];
  reasoning: ReasoningCapability[];
  multimodal: MultimodalCapability[];
}

export interface NLPCapability {
  id: string;
  name: string;
  type: 'sentiment-analysis' | 'text-classification' | 'named-entity-recognition' | 'summarization';
  accuracy: number;
  languages: string[];
  isActive: boolean;
}

export interface ComputerVisionCapability {
  id: string;
  name: string;
  type: 'object-detection' | 'image-classification' | 'face-recognition' | 'ocr';
  accuracy: number;
  supportedFormats: string[];
  isActive: boolean;
}

export interface ReasoningCapability {
  id: string;
  name: string;
  type: 'logical' | 'symbolic' | 'quantum';
  complexity: 'basic' | 'intermediate' | 'advanced';
  isActive: boolean;
}

export interface MultimodalCapability {
  id: string;
  name;
  name: string;
  inputTypes: string[];
  outputTypes: string[];
  fusionStrategy: string;
  isActive: boolean;
}

// API REST module interfaces
export interface APIRESTModule extends BaseModule {
  endpoints: APIEndpoint[];
  authentication: AuthenticationConfig;
  rateLimiting: RateLimitConfig;
  cors: CORSConfig;
  documentation: DocumentationConfig;
}

export interface APIEndpoint {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  description: string;
  parameters: Parameter[];
  responses: Response[];
  rateLimit?: number;
  requiresAuth: boolean;
}

export interface Parameter {
  name: string;
  type: string;
  required: boolean;
  description: string;
  defaultValue?: unknown;
}

export interface Response {
  code: number;
  description: string;
  schema: Record<string, unknown>;
}

export interface AuthenticationConfig {
  enabled: boolean;
  methods: ('api-key' | 'jwt' | 'oauth2')[];
  apiKeyHeader: string;
  jwtSecret?: string;
  oauthProviders: OAuthProvider[];
}

export interface OAuthProvider {
  name: string;
  clientId: string;
  clientSecret: string;
  authorizationUrl: string;
  tokenUrl: string;
}

export interface RateLimitConfig {
  enabled: boolean;
  requestsPerMinute: number;
  burstSize: number;
  strategy: 'fixed-window' | 'sliding-window' | 'token-bucket';
}

export interface CORSConfig {
  enabled: boolean;
  allowedOrigins: string[];
  allowedMethods: string[];
  allowedHeaders: string[];
  allowCredentials: boolean;
}

export interface DocumentationConfig {
  swaggerEnabled: boolean;
  redocEnabled: boolean;
  customDocs?: string;
}

// Security module interfaces
export interface SecurityModule extends BaseModule {
  encryption: EncryptionConfig;
  authentication: SecurityAuthenticationConfig;
  authorization: AuthorizationConfig;
  audit: AuditConfig;
  threatDetection: ThreatDetectionConfig;
}

export interface EncryptionConfig {
  algorithm: string;
  keySize: number;
  keyRotationEnabled: boolean;
  rotationInterval: number;
}

export interface SecurityAuthenticationConfig {
  methods: string[];
  mfaEnabled: boolean;
  sessionTimeout: number;
  maxLoginAttempts: number;
}

export interface AuthorizationConfig {
  rbacEnabled: boolean;
  permissions: Permission[];
  roles: Role[];
}

export interface Permission {
  id: string;
  name: string;
  resource: string;
  action: string;
}

export interface Role {
  id: string;
  name: string;
  permissions: string[];
  description: string;
}

export interface AuditConfig {
  enabled: boolean;
  logLevel: 'info' | 'warning' | 'error';
  retentionPeriod: number;
  sensitiveActions: string[];
}

export interface ThreatDetectionConfig {
  enabled: boolean;
  rules: ThreatRule[];
  autoResponse: boolean;
  notificationChannels: string[];
}

export interface ThreatRule {
  id: string;
  name: string;
  pattern: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  action: string;
}

// Distributed processing module interfaces
export interface DistributedProcessingModule extends BaseModule {
  nodes: Node[];
  loadBalancing: LoadBalancingConfig;
  faultTolerance: FaultToleranceConfig;
  autoScaling: AutoScalingConfig;
  taskScheduling: TaskSchedulingConfig;
}

export interface Node {
  id: string;
  hostname: string;
  ip: string;
  status: 'online' | 'offline' | 'maintenance';
  resources: NodeResources;
  lastHeartbeat: Date;
}

export interface NodeResources {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
}

export interface LoadBalancingConfig {
  strategy: 'round-robin' | 'least-connections' | 'adaptive' | 'weighted';
  healthCheckEnabled: boolean;
  healthCheckInterval: number;
  failoverEnabled: boolean;
}

export interface FaultToleranceConfig {
  circuitBreakerEnabled: boolean;
  replicationFactor: number;
  checkpointingEnabled: boolean;
  recoveryStrategy: string;
}

export interface AutoScalingConfig {
  enabled: boolean;
  minNodes: number;
  maxNodes: number;
  scaleUpThreshold: number;
  scaleDownThreshold: number;
  cooldownPeriod: number;
}

export interface TaskSchedulingConfig {
  strategy: 'fifo' | 'priority' | 'fair' | 'deadline';
  preemptionEnabled: boolean;
  resourceConstraints: boolean;
  retryPolicy: RetryPolicy;
}

export interface RetryPolicy {
  maxRetries: number;
  backoffStrategy: 'fixed' | 'exponential' | 'linear';
  initialDelay: number;
  maxDelay: number;
}

// System configuration interfaces
export interface SystemConfig {
  app: AppConfig;
  database: DatabaseConfig;
  cache: CacheConfig;
  security: SecurityConfig;
  monitoring: MonitoringConfig;
  api: APIConfig;
}

export interface AppConfig {
  name: string;
  version: string;
  environment: 'development' | 'staging' | 'production';
  debug: boolean;
  logLevel: string;
}

export interface DatabaseConfig {
  host: string;
  port: number;
  name: string;
  username: string;
  password: string;
  poolSize: number;
  ssl: boolean;
}

export interface CacheConfig {
  type: string;
  host: string;
  port: number;
  password?: string;
  maxMemory: number;
  ttl: number;
}

export interface SecurityConfig {
  jwtSecret: string;
  bcryptRounds: number;
  sessionTimeout: number;
  corsOrigins: string[];
}

export interface MonitoringConfig {
  enabled: boolean;
  interval: number;
  metricsRetention: number;
  alertChannels: string[];
}

export interface APIConfig {
  host: string;
  port: number;
  corsEnabled: boolean;
  rateLimitEnabled: boolean;
  docsEnabled: boolean;
}

// Error handling interfaces
export interface BlazeAIError extends Error {
  code: string;
  statusCode: number;
  details?: Record<string, unknown>;
  timestamp: Date;
  requestId?: string;
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
    timestamp: string;
    requestId?: string;
  };
}

// Response interfaces
export interface APIResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  requestId?: string;
}

export interface PaginatedResponse<T = unknown> extends APIResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

// Navigation interfaces
export interface NavigationProps {
  navigation: any;
  route: any;
}

export interface TabNavigationProps {
  navigation: any;
  route: any;
}

// Theme interfaces
export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    warning: string;
    success: string;
    info: string;
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
    xxl: number;
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
  typography: {
    h1: TextStyle;
    h2: TextStyle;
    h3: TextStyle;
    h4: TextStyle;
    h5: TextStyle;
    h6: TextStyle;
    body: TextStyle;
    caption: TextStyle;
    button: TextStyle;
  };
}

export interface TextStyle {
  fontSize: number;
  fontWeight: 'normal' | 'bold' | '100' | '200' | '300' | '400' | '500' | '600' | '700' | '800' | '900';
  lineHeight: number;
  letterSpacing?: number;
}

// Component props interfaces
export interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: string;
}

export interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  onPress?: () => void;
  variant?: 'default' | 'elevated' | 'outlined';
}

export interface InputProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChangeText: (text: string) => void;
  error?: string;
  disabled?: boolean;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad';
}

export interface ModalProps {
  visible: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  showCloseButton?: boolean;
}


