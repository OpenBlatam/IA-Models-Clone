/**
 * Navigation types and interfaces
 */

export interface AnalysisParams {
  imageUri?: string;
  videoUri?: string;
  analysis?: Record<string, unknown>;
  fromHistory?: boolean;
}

export interface RecommendationsParams {
  recommendations?: Record<string, unknown>;
  analysis?: Record<string, unknown>;
}

export interface ReportParams {
  analysis: Record<string, unknown>;
}

export type RootStackParamList = {
  MainTabs: undefined;
  RealTimeScan: undefined;
  Analysis: AnalysisParams;
  Recommendations: RecommendationsParams;
  Report: ReportParams;
  Comparison: undefined;
};

export type TabParamList = {
  Home: undefined;
  Camera: undefined;
  History: undefined;
  Profile: undefined;
};

declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}

