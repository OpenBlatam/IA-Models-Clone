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

