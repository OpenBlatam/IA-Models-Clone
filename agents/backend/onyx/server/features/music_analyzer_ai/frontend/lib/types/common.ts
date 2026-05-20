/**
 * Common TypeScript types and interfaces.
 * Shared types used across the application.
 */

/**
 * Generic API response wrapper.
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

/**
 * Pagination metadata.
 */
export interface PaginationMeta {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

/**
 * Paginated response.
 */
export interface PaginatedResponse<T> {
  data: T[];
  meta: PaginationMeta;
}

/**
 * Sort configuration.
 */
export interface SortConfig {
  field: string;
  order: 'asc' | 'desc';
}

/**
 * Filter configuration.
 */
export interface FilterConfig {
  [key: string]: unknown;
}

/**
 * View mode options.
 */
export type ViewMode = 'grid' | 'list' | 'compact';

/**
 * Loading state.
 */
export interface LoadingState {
  isLoading: boolean;
  error: Error | null;
}

/**
 * Async operation state.
 */
export interface AsyncState<T> extends LoadingState {
  data: T | null;
}

/**
 * Result type for operations that can fail.
 */
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Optional value type.
 */
export type Optional<T> = T | null | undefined;

/**
 * Non-nullable type.
 */
export type NonNullable<T> = T extends null | undefined ? never : T;

/**
 * Deep readonly type.
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

/**
 * Deep partial type.
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Make specific properties required.
 */
export type RequireFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * Make specific properties optional.
 */
export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Extract promise type.
 */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/**
 * Function type helper.
 */
export type FunctionType<T extends (...args: never[]) => unknown> = T;

/**
 * Event handler type.
 */
export type EventHandler<T = unknown> = (event: T) => void;

/**
 * Async function type.
 */
export type AsyncFunction<T extends unknown[] = never[], R = unknown> = (
  ...args: T
) => Promise<R>;

