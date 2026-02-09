/**
 * Result Pattern for functional error handling
 * Pattern: Functional approach to error handling without exceptions
 */

export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export const Result = {
  ok: <T>(data: T): Result<T, never> => ({ success: true, data }),
  err: <E>(error: E): Result<never, E> => ({ success: false, error }),
  
  map: <T, U, E>(
    result: Result<T, E>,
    fn: (data: T) => U
  ): Result<U, E> => {
    if (result.success) {
      return Result.ok(fn(result.data));
    }
    return result;
  },
  
  mapErr: <T, E, F>(
    result: Result<T, E>,
    fn: (error: E) => F
  ): Result<T, F> => {
    if (!result.success) {
      return Result.err(fn(result.error));
    }
    return result;
  },
  
  unwrap: <T, E>(result: Result<T, E>): T => {
    if (result.success) {
      return result.data;
    }
    throw result.error;
  },
  
  unwrapOr: <T, E>(result: Result<T, E>, defaultValue: T): T => {
    if (result.success) {
      return result.data;
    }
    return defaultValue;
  },
  
  isOk: <T, E>(result: Result<T, E>): result is { success: true; data: T } => {
    return result.success;
  },
  
  isErr: <T, E>(result: Result<T, E>): result is { success: false; error: E } => {
    return !result.success;
  },
};






