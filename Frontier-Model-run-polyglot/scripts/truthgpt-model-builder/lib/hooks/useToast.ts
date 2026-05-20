/**
 * Hook para Sistema de Toast
 * ===========================
 * 
 * Hook para gestionar notificaciones toast
 */

import { useState, useCallback } from 'react'
import { Toast, ToastType } from '@/components/Toast'

export interface UseToastResult {
  toasts: Toast[]
  showToast: (type: ToastType, title: string, message?: string, options?: Partial<Toast>) => string
  success: (title: string, message?: string) => string
  error: (title: string, message?: string) => string
  warning: (title: string, message?: string) => string
  info: (title: string, message?: string) => string
  loading: (title: string, message?: string) => string
  dismiss: (id: string) => void
  dismissAll: () => void
}

/**
 * Hook para usar el sistema de toast
 */
export function useToast(): UseToastResult {
  const [toasts, setToasts] = useState<Toast[]>([])

  const showToast = useCallback((
    type: ToastType,
    title: string,
    message?: string,
    options?: Partial<Toast>
  ): string => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    const toast: Toast = {
      id,
      type,
      title,
      message,
      duration: 5000,
      ...options
    }

    setToasts(prev => [...prev, toast])
    return id
  }, [])

  const success = useCallback((title: string, message?: string) => {
    return showToast('success', title, message)
  }, [showToast])

  const error = useCallback((title: string, message?: string) => {
    return showToast('error', title, message, { duration: 7000 })
  }, [showToast])

  const warning = useCallback((title: string, message?: string) => {
    return showToast('warning', title, message)
  }, [showToast])

  const info = useCallback((title: string, message?: string) => {
    return showToast('info', title, message)
  }, [showToast])

  const loading = useCallback((title: string, message?: string) => {
    return showToast('loading', title, message, { duration: 0 })
  }, [showToast])

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }, [])

  const dismissAll = useCallback(() => {
    setToasts([])
  }, [])

  return {
    toasts,
    showToast,
    success,
    error,
    warning,
    info,
    loading,
    dismiss,
    dismissAll
  }
}







