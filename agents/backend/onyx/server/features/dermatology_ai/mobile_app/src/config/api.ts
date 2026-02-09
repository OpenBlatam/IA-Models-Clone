// API Configuration
export const API_BASE_URL = __DEV__ 
  ? 'http://localhost:8006' 
  : 'https://your-production-api.com';

export const API_ENDPOINTS = {
  // Analysis
  ANALYZE_IMAGE: '/dermatology/analyze-image',
  ANALYZE_VIDEO: '/dermatology/analyze-video',
  ADVANCED_IMAGE_ANALYSIS: '/dermatology/image-analysis/advanced',
  ADVANCED_VIDEO_ANALYSIS: '/dermatology/video-analysis/advanced',
  
  // Recommendations
  GET_RECOMMENDATIONS: '/dermatology/get-recommendations',
  INTELLIGENT_RECOMMENDATIONS: '/dermatology/recommendations/intelligent',
  SMART_RECOMMENDATIONS: '/dermatology/smart-recommendations/generate',
  ML_RECOMMENDATIONS: '/dermatology/ml-recommendations/generate',
  
  // History & Progress
  HISTORY: '/dermatology/history',
  COMPARE_HISTORY: '/dermatology/history/compare',
  TIMELINE: '/dermatology/history/timeline',
  PROGRESS_ANALYZE: '/dermatology/progress/analyze',
  PROGRESS_TIMELINE: '/dermatology/progress/timeline',
  
  // Reports
  REPORT_JSON: '/dermatology/report/json',
  REPORT_PDF: '/dermatology/report/pdf',
  REPORT_HTML: '/dermatology/report/html',
  ADVANCED_REPORT: '/dermatology/reports/advanced',
  
  // Analytics
  ANALYTICS_USER: '/dermatology/analytics/user',
  ANALYTICS_PROGRESS: '/dermatology/analytics/progress',
  STATISTICS: '/dermatology/statistics',
  
  // Products
  PRODUCTS_SEARCH: '/dermatology/products/search',
  PRODUCTS_COMPARE: '/dermatology/products/compare',
  PRODUCTS_TRACK: '/dermatology/products/track',
  PRODUCTS_INSIGHTS: '/dermatology/products/insights',
  
  // Health
  HEALTH: '/dermatology/health',
  HEALTH_DETAILED: '/dermatology/health/detailed',
  
  // Body Area
  ANALYZE_BODY_AREA: '/dermatology/analyze-body-area',
  
  // Alerts
  ALERTS: '/dermatology/alerts',
  ALERTS_SUMMARY: '/dermatology/alerts/summary',
  INTELLIGENT_ALERTS: '/dermatology/alerts/intelligent',
  
  // Goals & Journal
  GOALS_CREATE: '/dermatology/goals/create',
  GOALS_USER: '/dermatology/goals/user',
  JOURNAL_ENTRY: '/dermatology/journal/entry',
  JOURNAL_USER: '/dermatology/journal/user',
  
  // ML & Predictions
  ML_PREDICT: '/dermatology/ml/predict',
  CONDITIONS_PREDICT: '/dermatology/conditions/predict',
  FUTURE_PREDICTION: '/dermatology/future-prediction/generate',
  
  // Visualization
  VISUALIZATION_RADAR: '/dermatology/visualization/radar',
  VISUALIZATION_TIMELINE: '/dermatology/visualization/timeline',
  VISUALIZATION_COMPARISON: '/dermatology/visualization/comparison',
} as const;

export default {
  API_BASE_URL,
  API_ENDPOINTS,
};

