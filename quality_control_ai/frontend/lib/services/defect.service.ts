import type { Defect } from '@/modules/inspection/types';

export class DefectService {
  static groupByType(defects: Defect[]): Record<string, number> {
    return defects.reduce(
      (acc, defect) => {
        acc[defect.type] = (acc[defect.type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }

  static groupBySeverity(defects: Defect[]): {
    critical: number;
    severe: number;
    moderate: number;
    minor: number;
  } {
    return defects.reduce(
      (acc, defect) => {
        acc[defect.severity] = (acc[defect.severity] || 0) + 1;
        return acc;
      },
      {
        critical: 0,
        severe: 0,
        moderate: 0,
        minor: 0,
      }
    );
  }

  static hasCriticalDefects(defects: Defect[]): boolean {
    return defects.some((defect) => defect.severity === 'critical');
  }

  static getTotalDefectArea(defects: Defect[]): number {
    return defects.reduce((total, defect) => total + defect.area, 0);
  }

  static getAverageConfidence(defects: Defect[]): number {
    if (defects.length === 0) return 0;
    const sum = defects.reduce((total, defect) => total + defect.confidence, 0);
    return sum / defects.length;
  }
}

