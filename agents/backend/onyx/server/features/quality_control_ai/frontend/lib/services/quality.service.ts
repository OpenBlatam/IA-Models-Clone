import { QUALITY_THRESHOLDS } from '@/config/constants';
import type { InspectionResult } from '@/modules/inspection/types';

export class QualityService {
  static calculateQualityScore(result: InspectionResult): number {
    let score = 100.0;

    result.anomalies.forEach((anomaly) => {
      if (anomaly.severity === 'high') {
        score -= 10.0;
      } else if (anomaly.severity === 'medium') {
        score -= 5.0;
      } else {
        score -= 2.0;
      }
    });

    result.defects.forEach((defect) => {
      if (defect.severity === 'critical') {
        score -= 20.0;
      } else if (defect.severity === 'severe') {
        score -= 15.0;
      } else if (defect.severity === 'moderate') {
        score -= 8.0;
      } else {
        score -= 3.0;
      }
    });

    return Math.max(0.0, Math.min(100.0, Math.round(score * 100) / 100));
  }

  static getQualityStatus(score: number): 'excellent' | 'good' | 'acceptable' | 'poor' | 'rejected' {
    if (score >= QUALITY_THRESHOLDS.EXCELLENT) return 'excellent';
    if (score >= QUALITY_THRESHOLDS.GOOD) return 'good';
    if (score >= QUALITY_THRESHOLDS.ACCEPTABLE) return 'acceptable';
    if (score >= QUALITY_THRESHOLDS.POOR) return 'poor';
    return 'rejected';
  }

  static getRecommendation(score: number, hasCriticalDefects: boolean): string {
    if (score >= QUALITY_THRESHOLDS.EXCELLENT) {
      return 'Producto aprobado - calidad excelente';
    }
    if (score >= QUALITY_THRESHOLDS.GOOD) {
      return 'Producto aprobado - calidad buena';
    }
    if (score >= QUALITY_THRESHOLDS.ACCEPTABLE) {
      return 'Producto aprobado con observaciones menores';
    }
    if (score >= QUALITY_THRESHOLDS.POOR) {
      return 'Producto requiere revisión manual';
    }
    if (hasCriticalDefects) {
      return 'Producto rechazado - defectos críticos detectados';
    }
    return 'Producto rechazado - múltiples defectos detectados';
  }
}

