export interface ApiError {
  message: string;
  code?: string;
  statusCode?: number;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

export interface LoadingState {
  isLoading: boolean;
  error: ApiError | null;
}

export interface AsyncState<T> extends LoadingState {
  data: T | null;
}

export interface SelectOption<T = string> {
  label: string;
  value: T;
  disabled?: boolean;
}

export interface ToastConfig {
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

export interface NavigationParams {
  [key: string]: string | number | boolean | undefined;
}

