/**
 * Error Boundary component for catching and displaying React errors.
 * Provides a fallback UI when an error occurs in the component tree.
 * Refactored with better error handling and accessibility.
 */

'use client';

import React, { Component, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { getErrorMessage } from '@/lib/errors';
import Link from 'next/link';
import { ROUTES } from '@/lib/constants';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  level?: 'page' | 'component';
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

/**
 * Error Boundary component that catches errors in child components.
 * Enhanced with better error reporting and recovery options.
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Log error to error reporting service
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Update state with error info
    this.setState({ errorInfo });

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // In production, send to error tracking service
    // Example: Sentry.captureException(error, { contexts: { react: errorInfo } });
  }

  /**
   * Handles error reset.
   */
  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  /**
   * Handles page reload.
   */
  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const errorMessage = getErrorMessage(this.state.error);
      const isPageLevel = this.props.level === 'page';

      return (
        <div
          className={`${
            isPageLevel ? 'min-h-screen' : 'min-h-[400px]'
          } flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4`}
          role="alert"
          aria-live="assertive"
        >
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-8 border border-white/20 max-w-md w-full">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle
                className="w-8 h-8 text-red-400 flex-shrink-0"
                aria-hidden="true"
              />
              <h2 className="text-2xl font-bold text-white">
                Algo salió mal
              </h2>
            </div>
            <p className="text-gray-300 mb-4">{errorMessage}</p>

            {this.state.error && (
              <details className="mt-4">
                <summary className="text-sm text-gray-400 cursor-pointer mb-2 hover:text-gray-300 transition-colors">
                  Detalles del error
                </summary>
                <pre className="text-xs text-gray-500 bg-black/20 p-3 rounded overflow-auto max-h-40">
                  {this.state.error.stack || this.state.error.toString()}
                </pre>
              </details>
            )}

            <div className="flex flex-col sm:flex-row gap-3 mt-6">
              <button
                onClick={this.handleReset}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400"
                type="button"
                aria-label="Intentar de nuevo"
              >
                <RefreshCw className="w-4 h-4" aria-hidden="true" />
                Intentar de nuevo
              </button>

              {isPageLevel && (
                <>
                  <button
                    onClick={this.handleReload}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400"
                    type="button"
                    aria-label="Recargar página"
                  >
                    <RefreshCw className="w-4 h-4" aria-hidden="true" />
                    Recargar
                  </button>
                  <Link
                    href={ROUTES.HOME}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400"
                    aria-label="Ir al inicio"
                  >
                    <Home className="w-4 h-4" aria-hidden="true" />
                    Inicio
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
