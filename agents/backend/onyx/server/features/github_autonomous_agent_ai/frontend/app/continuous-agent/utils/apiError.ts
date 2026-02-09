/**
 * API error handling utilities
 * 
 * Provides consistent error message extraction and handling
 */

/**
 * Type guard to check if error has a message property
 */
const isErrorWithMessage = (error: unknown): error is { message: string } => {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as { message: unknown }).message === "string"
  );
};

/**
 * Type guard to check if error has a detail property
 */
const isErrorWithDetail = (error: unknown): error is { detail: string } => {
  return (
    typeof error === "object" &&
    error !== null &&
    "detail" in error &&
    typeof (error as { detail: unknown }).detail === "string"
  );
};

/**
 * Extracts a user-friendly error message from various error types
 * @param error - The error to extract message from
 * @param defaultMessage - Default message if extraction fails
 * @returns User-friendly error message
 */
export const getApiErrorMessage = (error: unknown, defaultMessage: string): string => {
  // Early returns for common error types
  if (error instanceof Error) {
    return error.message || defaultMessage;
  }

  if (typeof error === "string") {
    return error || defaultMessage;
  }

  if (isErrorWithMessage(error)) {
    return error.message || defaultMessage;
  }

  if (isErrorWithDetail(error)) {
    return error.detail || defaultMessage;
  }

  // Fallback to default message
  return defaultMessage;
};

/**
 * Handles API errors by extracting message and throwing
 * @param error - The error to handle
 * @param defaultMessage - Default message if extraction fails
 * @throws Error with user-friendly message
 */
export const handleApiError = (error: unknown, defaultMessage: string): never => {
  const message = getApiErrorMessage(error, defaultMessage);
  console.error("API Error:", error);
  throw new Error(message);
};

/**
 * Checks if an error is a network error
 * @param error - The error to check
 * @returns True if error appears to be a network error
 */
export const isNetworkError = (error: unknown): boolean => {
  if (error instanceof Error) {
    return (
      error.message.includes("fetch") ||
      error.message.includes("network") ||
      error.message.includes("NetworkError") ||
      error.message.includes("Failed to fetch")
    );
  }
  return false;
};

/**
 * Checks if an error is a timeout error
 * @param error - The error to check
 * @returns True if error appears to be a timeout
 */
export const isTimeoutError = (error: unknown): boolean => {
  if (error instanceof Error) {
    return (
      error.message.includes("timeout") ||
      error.message.includes("Timeout") ||
      error.message.includes("timed out")
    );
  }
  return false;
};




