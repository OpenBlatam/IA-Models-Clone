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

export interface BatchInspectionRequest {
  images: string[];
}

