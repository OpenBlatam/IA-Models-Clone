"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { AppError, logError } from "@/lib/error-handling";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showRetry?: boolean;
  retryText?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  retryCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  private retryTimeoutId: NodeJS.Timeout | null = null;

  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, retryCount: 0 };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, retryCount: 0 };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const appError = new AppError(
      `Component error: ${error.message}`,
      500,
      true,
      {
        componentStack: errorInfo.componentStack,
        errorBoundary: true,
        retryCount: this.state.retryCount
      }
    );

    logError(appError, {
      component: 'ErrorBoundary',
      errorInfo: errorInfo.componentStack
    });

    this.props.onError?.(error, errorInfo);
  }

  handleRetry = () => {
    const newRetryCount = this.state.retryCount + 1;
    
    if (newRetryCount > 3) {
      console.warn('Max retry attempts reached for ErrorBoundary');
      return;
    }

    this.setState({ 
      hasError: false, 
      error: null, 
      retryCount: newRetryCount 
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Card className="p-6 m-4 border-destructive">
          <div className="flex items-center space-x-4">
            <AlertTriangle className="h-8 w-8 text-destructive" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-destructive">
                Algo salió mal
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                {this.state.error?.message || 'Ha ocurrido un error inesperado'}
              </p>
              {this.state.retryCount > 0 && (
                <p className="text-xs text-muted-foreground mt-1">
                  Intentos de recuperación: {this.state.retryCount}/3
                </p>
              )}
            </div>
          </div>
          
          {this.props.showRetry !== false && this.state.retryCount < 3 && (
            <div className="mt-4 flex space-x-2">
              <Button 
                onClick={this.handleRetry}
                variant="outline"
                size="sm"
                className="flex items-center space-x-2"
              >
                <RefreshCw className="h-4 w-4" />
                <span>{this.props.retryText || 'Reintentar'}</span>
              </Button>
              <Button 
                onClick={() => window.location.reload()}
                variant="outline"
                size="sm"
              >
                Recargar página
              </Button>
            </div>
          )}
        </Card>
      );
    }

    return this.props.children;
  }
}
