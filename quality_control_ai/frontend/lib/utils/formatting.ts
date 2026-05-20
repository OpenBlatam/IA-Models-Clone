import { QUALITY_THRESHOLDS, QUALITY_STATUS_LABELS } from '@/config/constants';
import { getSmartDate } from './date';

export const formatQualityScore = (score: number): string => {
  return `${score.toFixed(1)}/100`;
};

export const getQualityColor = (score: number): string => {
  if (score >= QUALITY_THRESHOLDS.EXCELLENT) return 'text-quality-excellent';
  if (score >= QUALITY_THRESHOLDS.GOOD) return 'text-quality-good';
  if (score >= QUALITY_THRESHOLDS.ACCEPTABLE) return 'text-quality-acceptable';
  if (score >= QUALITY_THRESHOLDS.POOR) return 'text-quality-poor';
  return 'text-quality-rejected';
};

export const getQualityBgColor = (score: number): string => {
  if (score >= QUALITY_THRESHOLDS.EXCELLENT) return 'bg-quality-excellent';
  if (score >= QUALITY_THRESHOLDS.GOOD) return 'bg-quality-good';
  if (score >= QUALITY_THRESHOLDS.ACCEPTABLE) return 'bg-quality-acceptable';
  if (score >= QUALITY_THRESHOLDS.POOR) return 'bg-quality-poor';
  return 'bg-quality-rejected';
};

export const getStatusLabel = (status: string): string => {
  return QUALITY_STATUS_LABELS[status as keyof typeof QUALITY_STATUS_LABELS] || status;
};

export const formatDate = (dateString: string): string => {
  return getSmartDate(dateString);
};

export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100)} ${sizes[i]}`;
};

