export const QUALITY_THRESHOLDS = {
  EXCELLENT: 90,
  GOOD: 75,
  ACCEPTABLE: 60,
  POOR: 40,
} as const;

export const INSPECTION_INTERVAL = 2000; // ms
export const CAMERA_REFRESH_INTERVAL = 2000; // ms
export const ALERTS_REFRESH_INTERVAL = 3000; // ms
export const STATISTICS_REFRESH_INTERVAL = 5000; // ms

export const MAX_HISTORY_ITEMS = 100;
export const MAX_ALERTS_ITEMS = 200;
export const MAX_DISPLAYED_ALERTS = 10;

export const DEFAULT_CAMERA_SETTINGS = {
  camera_index: 0,
  resolution_width: 1920,
  resolution_height: 1080,
  fps: 30,
  brightness: 0.5,
  contrast: 0.5,
  saturation: 0.5,
  exposure: 0.5,
  auto_focus: true,
  white_balance: 'auto',
} as const;

export const DEFAULT_DETECTION_SETTINGS = {
  confidence_threshold: 0.5,
  nms_threshold: 0.4,
  anomaly_threshold: 0.7,
  use_autoencoder: true,
  use_statistical: true,
  object_detection_model: 'yolov8',
  min_defect_size: 10,
  max_defect_size: 10000,
} as const;

export const QUALITY_STATUS_LABELS = {
  excellent: 'Excellent',
  good: 'Good',
  acceptable: 'Acceptable',
  poor: 'Poor',
  rejected: 'Rejected',
} as const;

export const SEVERITY_COLORS = {
  critical: 'text-red-700 bg-red-50 border-red-200',
  severe: 'text-orange-700 bg-orange-50 border-orange-200',
  moderate: 'text-yellow-700 bg-yellow-50 border-yellow-200',
  minor: 'text-green-700 bg-green-50 border-green-200',
} as const;

export const ALERT_LEVEL_COLORS = {
  critical: 'text-red-700 bg-red-50 border-red-200',
  error: 'text-red-600 bg-red-50 border-red-200',
  warning: 'text-yellow-700 bg-yellow-50 border-yellow-200',
  info: 'text-blue-700 bg-blue-50 border-blue-200',
} as const;

