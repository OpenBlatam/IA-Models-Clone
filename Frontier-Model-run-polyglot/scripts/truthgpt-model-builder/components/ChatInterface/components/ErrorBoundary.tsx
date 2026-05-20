/**
 * ErrorBoundary Component
 * Catches and displays errors gracefully
 */

'use client'

import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'
import { handleError, getUserFriendlyError } from '../utils/errorHandling'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
    }
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    handleError(error, 'ErrorBoundary')
    
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
    })
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="error-boundary">
          <div className="error-boundary__content">
            <AlertCircle size={48} className="error-boundary__icon" />
            <h2 className="error-boundary__title">Algo salió mal</h2>
            <p className="error-boundary__message">
              {this.state.error
                ? getUserFriendlyError(this.state.error)
                : 'Ha ocurrido un error inesperado'}
            </p>
            <button
              type="button"
              onClick={this.handleReset}
              className="error-boundary__button"
            >
              <RefreshCw size={16} />
              Intentar de nuevo
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary




