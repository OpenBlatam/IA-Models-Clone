"use client";

import { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class AgentsErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("AgentsErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="p-4 rounded-lg bg-red-50 border border-red-200">
            <h3 className="text-sm font-semibold text-red-800 mb-2">
              Error al cargar agentes
            </h3>
            <p className="text-xs text-red-600 mb-3">
              {this.state.error?.message || "Ocurrió un error inesperado"}
            </p>
            <button
              onClick={() => this.setState({ hasError: false, error: undefined })}
              className="px-3 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
            >
              Reintentar
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}








