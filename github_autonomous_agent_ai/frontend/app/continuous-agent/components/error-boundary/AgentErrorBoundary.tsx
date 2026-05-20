"use client";

import React from "react";
import { ErrorBoundary } from "react-error-boundary";
import { Button } from "../ui/Button";
import { ErrorAlert } from "../ui/ErrorAlert";
import { ERROR_MESSAGES } from "../../constants/messages";

type ErrorFallbackProps = {
  readonly error: Error;
  readonly resetErrorBoundary: () => void;
};

/**
 * Fallback UI component for error boundary
 */
const ErrorFallback = ({ error, resetErrorBoundary }: ErrorFallbackProps): JSX.Element => {
  return (
    <div
      className="flex flex-col items-center justify-center p-8 space-y-4"
      role="alert"
      aria-live="assertive"
    >
      <ErrorAlert message={error.message || ERROR_MESSAGES.LOAD_AGENTS} />
      <Button
        type="button"
        variant="primary"
        onClick={resetErrorBoundary}
        ariaLabel="Reintentar"
      >
        Reintentar
      </Button>
    </div>
  );
};

type AgentErrorBoundaryProps = {
  readonly children: React.ReactNode;
  readonly fallback?: React.ComponentType<ErrorFallbackProps>;
  readonly onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  readonly onReset?: () => void;
};

/**
 * Error Boundary component for Continuous Agent feature
 * 
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI instead of crashing.
 * 
 * Features:
 * - Better error handling with reset functionality
 * - Customizable fallback UI
 * - Error logging support
 * - Automatic error recovery
 * 
 * @example
 * ```tsx
 * <AgentErrorBoundary onError={(error) => console.error(error)}>
 *   <AgentCard agent={agent} />
 * </AgentErrorBoundary>
 * ```
 */
export const AgentErrorBoundary = ({
  children,
  fallback = ErrorFallback,
  onError,
  onReset,
}: AgentErrorBoundaryProps): JSX.Element => {
  const handleError = (error: Error, errorInfo: { componentStack: string }) => {
    // Import logger dynamically to avoid circular dependencies
    import("../../utils/logger").then(({ logError }) => {
      logError("AgentErrorBoundary caught an error", error, {
        componentStack: errorInfo.componentStack,
      });
    });

    // Call optional error handler
    onError?.(error, {
      componentStack: errorInfo.componentStack,
    } as React.ErrorInfo);
  };

  return (
    <ErrorBoundary
      FallbackComponent={fallback}
      onError={handleError}
      onReset={onReset}
      resetKeys={[Date.now()]} // Reset when key changes
    >
      {children}
    </ErrorBoundary>
  );
};




