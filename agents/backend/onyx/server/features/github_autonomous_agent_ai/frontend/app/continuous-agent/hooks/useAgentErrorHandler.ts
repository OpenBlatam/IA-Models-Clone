import { useCallback } from "react";
import { toast } from "sonner";
import { SUCCESS_MESSAGES, ERROR_MESSAGES } from "../constants/messages";
import { getApiErrorMessage } from "../utils/apiError";
import { logError } from "../utils/logger";

/**
 * Options for configuring the error handler
 */
type AgentErrorHandlerOptions = {
  /** Callback invoked on successful operations */
  readonly onSuccess?: (message: string) => void;
  /** Callback invoked on errors */
  readonly onError?: (message: string) => void;
};

/**
 * Custom hook for handling agent-related errors and success messages
 * 
 * Provides:
 * - Centralized error handling with user-friendly messages
 * - Success notifications
 * - Type-safe async operation wrappers
 * 
 * @param options - Configuration options
 * @returns Object with error handling methods
 */
export const useAgentErrorHandler = (options: AgentErrorHandlerOptions = {}) => {
  const showSuccess = useCallback(
    (message: string): void => {
      toast.success(message);
      options.onSuccess?.(message);
    },
    [options]
  );

  const showError = useCallback(
    (message: string): void => {
      toast.error(message);
      options.onError?.(message);
    },
    [options]
  );

  const handleAsyncOperation = useCallback(
    async <T>(
      operation: () => Promise<T>,
      successMessage: string,
      errorMessage: string
    ): Promise<T | null> => {
      if (typeof operation !== "function") {
        const message = "Invalid operation function";
        showError(message);
        options.onError?.(message);
        return null;
      }

      try {
        const result = await operation();
        showSuccess(successMessage);
        return result;
      } catch (error) {
        const message = getApiErrorMessage(error, errorMessage);
        showError(message);
        
        // Log error
        logError("Operation failed", error instanceof Error ? error : new Error(String(error)), {
          operation: "handleAsyncOperation",
          errorMessage: message,
        });
        
        options.onError?.(message);
        return null;
      }
    },
    [showSuccess, showError, options]
  );

  const handleCreateAgent = useCallback(
    async <T>(operation: () => Promise<T>): Promise<T | null> => {
      return handleAsyncOperation(
        operation,
        SUCCESS_MESSAGES.AGENT_CREATED,
        ERROR_MESSAGES.CREATE_AGENT
      );
    },
    [handleAsyncOperation]
  );

  const handleToggleAgent = useCallback(
    async <T>(
      operation: () => Promise<T>,
      isActive: boolean
    ): Promise<T | null> => {
      const successMessage = isActive
        ? SUCCESS_MESSAGES.AGENT_ACTIVATED
        : SUCCESS_MESSAGES.AGENT_DEACTIVATED;
      return handleAsyncOperation(operation, successMessage, ERROR_MESSAGES.TOGGLE_AGENT);
    },
    [handleAsyncOperation]
  );

  const handleDeleteAgent = useCallback(
    async <T>(operation: () => Promise<T>): Promise<T | null> => {
      return handleAsyncOperation(
        operation,
        SUCCESS_MESSAGES.AGENT_DELETED,
        ERROR_MESSAGES.DELETE_AGENT
      );
    },
    [handleAsyncOperation]
  );

  const handleUpdateAgent = useCallback(
    async <T>(operation: () => Promise<T>): Promise<T | null> => {
      return handleAsyncOperation(
        operation,
        SUCCESS_MESSAGES.AGENT_UPDATED,
        ERROR_MESSAGES.UPDATE_AGENT
      );
    },
    [handleAsyncOperation]
  );

  return {
    showSuccess,
    showError,
    handleAsyncOperation,
    handleCreateAgent,
    handleToggleAgent,
    handleDeleteAgent,
    handleUpdateAgent,
  };
};


