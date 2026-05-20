/**
 * Hook useConfirm
 * ===============
 * 
 * Hook para manejar diálogos de confirmación
 */

import { useState, useCallback } from 'react'
import ConfirmDialog, { ConfirmDialogProps } from '@/components/ConfirmDialog'

export interface ConfirmOptions {
  title: string
  message: string
  type?: 'danger' | 'warning' | 'info' | 'success'
  confirmText?: string
  cancelText?: string
}

export function useConfirm() {
  const [dialog, setDialog] = useState<{
    isOpen: boolean
    options: ConfirmOptions | null
    onConfirm: (() => void) | null
    onCancel: (() => void) | null
    isLoading: boolean
  }>({
    isOpen: false,
    options: null,
    onConfirm: null,
    onCancel: null,
    isLoading: false
  })

  const confirm = useCallback(
    (options: ConfirmOptions): Promise<boolean> => {
      return new Promise((resolve) => {
        setDialog({
          isOpen: true,
          options,
          onConfirm: () => {
            setDialog(prev => ({ ...prev, isOpen: false, isLoading: false }))
            resolve(true)
          },
          onCancel: () => {
            setDialog(prev => ({ ...prev, isOpen: false, isLoading: false }))
            resolve(false)
          },
          isLoading: false
        })
      })
    },
    []
  )

  const setLoading = useCallback((isLoading: boolean) => {
    setDialog(prev => ({ ...prev, isLoading }))
  }, [])

  const close = useCallback(() => {
    setDialog(prev => {
      if (prev.onCancel) {
        prev.onCancel()
      }
      return {
        ...prev,
        isOpen: false,
        isLoading: false
      }
    })
  }, [])

  const ConfirmComponent = dialog.options ? (
    <ConfirmDialog
      isOpen={dialog.isOpen}
      onClose={close}
      onConfirm={() => {
        if (dialog.onConfirm) {
          dialog.onConfirm()
        }
      }}
      title={dialog.options.title}
      message={dialog.options.message}
      type={dialog.options.type}
      confirmText={dialog.options.confirmText}
      cancelText={dialog.options.cancelText}
      isLoading={dialog.isLoading}
    />
  ) : null

  return {
    confirm,
    setLoading,
    ConfirmComponent
  }
}

