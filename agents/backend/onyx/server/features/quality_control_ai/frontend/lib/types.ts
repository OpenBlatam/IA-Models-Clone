export interface InspectionResult {
  success: boolean;
  timestamp: string;
  quality_score: number;
  objects_detected: number;
  anomalies_detected: number;
  defects_detected: number;
  objects: DetectedObject[];
  anomalies: Anomaly[];
  defects: Defect[];
  summary: InspectionSummary;
  error?: string;
}

export interface DetectedObject {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  center: [number, number];
  area: number;
}

export interface Anomaly {
  type: string;
  confidence: number;
  location: [number, number];
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export interface Defect {
  type: DefectType;
  confidence: number;
  location: [number, number];
  severity: 'minor' | 'moderate' | 'severe' | 'critical';
  area: number;
  description: string;
}

export type DefectType =
  | 'Scratch'
  | 'Crack'
  | 'Dent'
  | 'Discoloration'
  | 'Deformation'
  | 'Missing Part'
  | 'Surface Imperfection'
  | 'Contamination'
  | 'Size Variation'
  | 'Other';

export interface InspectionSummary {
  status: 'excellent' | 'good' | 'acceptable' | 'poor' | 'rejected';
  quality_score: number;
  total_objects: number;
  total_anomalies: number;
  total_defects: number;
  defect_counts: Record<string, number>;
  severity_counts: {
    critical: number;
    severe: number;
    moderate: number;
    minor: number;
  };
  has_critical_defects: boolean;
  recommendation: string;
}

export interface CameraInfo {
  status: string;
  streaming: boolean;
  camera_index: number;
  resolution: {
    width: number;
    height: number;
  };
  fps: number;
  brightness: number;
  contrast: number;
  saturation: number;
  exposure: number;
}

export interface CameraSettings {
  camera_index?: number;
  resolution_width?: number;
  resolution_height?: number;
  fps?: number;
  brightness?: number;
  contrast?: number;
  saturation?: number;
  exposure?: number;
  auto_focus?: boolean;
  white_balance?: string;
}

export interface DetectionSettings {
  confidence_threshold?: number;
  nms_threshold?: number;
  anomaly_threshold?: number;
  use_autoencoder?: boolean;
  use_statistical?: boolean;
  object_detection_model?: string;
  min_defect_size?: number;
  max_defect_size?: number;
}

export interface Alert {
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  source: string;
  data?: Record<string, unknown>;
}

export interface AlertStatistics {
  total: number;
  by_level: {
    info: number;
    warning: number;
    error: number;
    critical: number;
  };
  recent_critical: number;
}

export interface BatchInspectionRequest {
  images: string[]; // Base64 encoded images
}

export interface ReportRequest {
  inspection_result: InspectionResult;
  format: 'json' | 'csv' | 'html';
  include_images?: boolean;
  include_charts?: boolean;
}

