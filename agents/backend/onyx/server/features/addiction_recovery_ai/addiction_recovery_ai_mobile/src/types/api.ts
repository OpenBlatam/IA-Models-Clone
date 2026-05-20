// API Error Types
export interface ApiError {
  detail: string;
  status_code: number;
}

// Generic API Response
export interface ApiResponse<T> {
  data?: T;
  error?: ApiError;
}

