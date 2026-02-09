import type { InspectionResult } from '../inspection/types';

export interface ReportRequest {
  inspection_result: InspectionResult;
  format: 'json' | 'csv' | 'html';
  include_images?: boolean;
  include_charts?: boolean;
}

