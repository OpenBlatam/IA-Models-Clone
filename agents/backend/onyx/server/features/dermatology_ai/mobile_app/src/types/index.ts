// API Types
export interface QualityScores {
  overall_score?: number;
  texture_score?: number;
  hydration_score?: number;
  elasticity_score?: number;
  pigmentation_score?: number;
  pore_size_score?: number;
  wrinkles_score?: number;
  redness_score?: number;
  dark_spots_score?: number;
}

export interface Condition {
  name: string;
  severity?: string;
  confidence?: number;
}

export interface AnalysisResult {
  id?: string;
  success?: boolean;
  quality_scores?: QualityScores;
  conditions?: Condition[];
  skin_type?: string;
  recommendations_priority?: string[];
  timestamp?: string;
  image_uri?: string;
  body_area?: string;
  analysis?: AnalysisResult;
}

export interface Product {
  id?: string;
  name: string;
  category?: string;
  description?: string;
  price?: number;
  benefits?: string[];
  image?: string;
}

export interface RoutineStep {
  name: string;
  description?: string;
  time?: string;
  products?: string[];
}

export interface Routine {
  steps: RoutineStep[];
  duration?: string;
  frequency?: string;
}

export interface Recommendations {
  products?: Product[];
  routine?: Routine;
  tips?: string[];
  warnings?: string[];
  priority?: string[];
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  image_uri?: string;
  video_uri?: string;
  quality_scores?: QualityScores;
  conditions?: Condition[];
  skin_type?: string;
  body_area?: string;
  analysis?: AnalysisResult;
}

export interface User {
  id: string;
  name?: string;
  email?: string;
  skin_type?: string;
  preferences?: Record<string, any>;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Redux Types
export interface AnalysisState {
  currentAnalysis: AnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
  recommendations: Recommendations | null;
}

export interface HistoryState {
  history: HistoryItem[];
  timeline: any[];
  isLoading: boolean;
  error: string | null;
}

export interface UserState {
  userId: string | null;
  userData: User | null;
  isAuthenticated: boolean;
}

export interface RootState {
  analysis: AnalysisState;
  history: HistoryState;
  user: UserState;
}

// Navigation Types
export type RootStackParamList = {
  MainTabs: undefined;
  RealTimeScan: undefined;
  Analysis: { imageUri?: string; videoUri?: string; analysis?: AnalysisResult; fromHistory?: boolean };
  Recommendations: { recommendations?: Recommendations; analysis?: AnalysisResult };
  Report: { analysis: AnalysisResult };
  Comparison: { 
    analysis1?: AnalysisResult; 
    analysis2?: AnalysisResult;
    recordId1?: string;
    recordId2?: string;
    label1?: string;
    label2?: string;
  };
};

export type TabParamList = {
  Home: undefined;
  Camera: undefined;
  History: undefined;
  Profile: undefined;
}

