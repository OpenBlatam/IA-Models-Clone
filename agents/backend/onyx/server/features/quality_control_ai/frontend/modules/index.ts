// Camera module
export * from './camera/types';
export { cameraApi } from './camera/api';
export { useCamera } from './camera/hooks/useCamera';
export { default as CameraView } from './camera/components/CameraView';
export { default as CameraSettingsModal } from './camera/components/CameraSettingsModal';

// Inspection module
export * from './inspection/types';
export { inspectionApi } from './inspection/api';
export { useInspection } from './inspection/hooks/useInspection';
export { default as InspectionResults } from './inspection/components/InspectionResults';
export { default as DefectList } from './inspection/components/DefectList';
export { default as ImageUpload } from './inspection/components/ImageUpload';

// Alerts module
export * from './alerts/types';
export { alertsApi } from './alerts/api';
export { useAlerts } from './alerts/hooks/useAlerts';
export { default as AlertsPanel } from './alerts/components/AlertsPanel';

// Detection module
export * from './detection/types';
export { detectionApi } from './detection/api';
export { default as DetectionSettingsModal } from './detection/components/DetectionSettingsModal';

// Reports module
export * from './reports/types';
export { reportsApi } from './reports/api';
export { default as ReportGenerator } from './reports/components/ReportGenerator';

// Statistics module
export { default as StatisticsPanel } from './statistics/components/StatisticsPanel';

// Control module
export { default as ControlPanel } from './control/components/ControlPanel';

