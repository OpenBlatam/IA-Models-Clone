/**
 * Common TypeScript Utility Types
 * Reusable type definitions and utilities
 */

/**
 * Makes all properties optional recursively
 */
export type DeepPartial<T> = T extends object
  ? {
      [P in keyof T]?: DeepPartial<T[P]>;
    }
  : T;

/**
 * Makes all properties required recursively
 */
export type DeepRequired<T> = T extends object
  ? {
      [P in keyof T]-?: DeepRequired<T[P]>;
    }
  : T;

/**
 * Makes all properties readonly recursively
 */
export type DeepReadonly<T> = T extends object
  ? {
      readonly [P in keyof T]: DeepReadonly<T[P]>;
    }
  : T;

/**
 * Extracts the value type from an object
 */
export type ValueOf<T> = T[keyof T];

/**
 * Creates a type from an array of strings
 */
export type ArrayToUnion<T extends readonly string[]> = T[number];

/**
 * Makes specific keys optional
 */
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Makes specific keys required
 */
export type Required<T, K extends keyof T> = T & { [P in K]-?: T[P] };

/**
 * Extracts the return type of a function
 */
export type ReturnType<T extends (...args: unknown[]) => unknown> = T extends (
  ...args: unknown[]
) => infer R
  ? R
  : never;

/**
 * Extracts the parameters of a function
 */
export type Parameters<T extends (...args: unknown[]) => unknown> = T extends (
  ...args: infer P
) => unknown
  ? P
  : never;

/**
 * Async function return type
 */
export type AsyncReturnType<T extends (...args: unknown[]) => Promise<unknown>> =
  T extends (...args: unknown[]) => Promise<infer R> ? R : never;

/**
 * Component props with children
 */
export interface ComponentWithChildren {
  children: React.ReactNode;
}

/**
 * Component props with className
 */
export interface ComponentWithClassName {
  className?: string;
}

/**
 * Component props with both children and className
 */
export interface ComponentProps extends ComponentWithChildren, ComponentWithClassName {}

/**
 * ID type (string or number)
 */
export type ID = string | number;

/**
 * Timestamp type
 */
export type Timestamp = string | Date | number;

/**
 * Status type for common statuses
 */
export type Status = 'idle' | 'loading' | 'success' | 'error';

/**
 * Pagination metadata
 */
export interface PaginationMeta {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  data: T[];
  meta: PaginationMeta;
}

/**
 * API response wrapper
 */
export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

/**
 * Error response
 */
export interface ApiError {
  message: string;
  code?: string;
  statusCode?: number;
  errors?: Record<string, string[]>;
}


