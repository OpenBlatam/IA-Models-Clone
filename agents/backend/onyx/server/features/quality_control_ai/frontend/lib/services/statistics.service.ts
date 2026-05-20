import type { InspectionResult } from '@/modules/inspection/types';

export class StatisticsService {
  static calculateAverageScore(results: InspectionResult[]): number {
    if (results.length === 0) return 0;
    const sum = results.reduce((total, result) => total + result.quality_score, 0);
    return Math.round((sum / results.length) * 100) / 100;
  }

  static getDefectTypeDistribution(results: InspectionResult[]): Record<string, number> {
    return results.reduce(
      (acc, result) => {
        result.defects.forEach((defect) => {
          acc[defect.type] = (acc[defect.type] || 0) + 1;
        });
        return acc;
      },
      {} as Record<string, number>
    );
  }

  static getSeverityDistribution(results: InspectionResult[]): {
    critical: number;
    severe: number;
    moderate: number;
    minor: number;
  } {
    return results.reduce(
      (acc, result) => {
        result.defects.forEach((defect) => {
          acc[defect.severity] = (acc[defect.severity] || 0) + 1;
        });
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

  static getRecentScores(results: InspectionResult[], limit = 10): Array<{
    score: number;
    defects: number;
  }> {
    return results.slice(0, limit).map((result) => ({
      score: result.quality_score,
      defects: result.defects_detected,
    }));
  }

  static getPassRate(results: InspectionResult[]): number {
    if (results.length === 0) return 0;
    const passed = results.filter((result) => result.quality_score >= 60).length;
    return Math.round((passed / results.length) * 10000) / 100;
  }
}

