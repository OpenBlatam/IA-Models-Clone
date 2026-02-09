'use client'

import { useCallback } from 'react'
import clsx from 'clsx'
import { AlertTriangle } from 'lucide-react'

interface ConfirmDialogProps {
  isOpen: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  variant?: 'danger' | 'warning' | 'info'
  onConfirm: () => void
  onCancel: () => void
}

const ConfirmDialog = ({
  isOpen,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'warning',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) => {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, action: () => void) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        action()
      }
    },
    []
  )

  const handleEscape = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel()
      }
    },
    [onCancel]
  )

  if (!isOpen) {
    return null
  }

  const variantClasses = {
    danger: 'bg-red-600 hover:bg-red-700',
    warning: 'bg-yellow-600 hover:bg-yellow-700',
    info: 'bg-primary-600 hover:bg-primary-700',
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      onClick={onCancel}
      onKeyDown={handleEscape}
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby="confirm-dialog-message"
      tabIndex={-1}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-4">
          <div
            className={clsx(
              'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center',
              variant === 'danger' && 'bg-red-100',
              variant === 'warning' && 'bg-yellow-100',
              variant === 'info' && 'bg-primary-100'
            )}
          >
            <AlertTriangle
              className={clsx(
                'w-6 h-6',
                variant === 'danger' && 'text-red-600',
                variant === 'warning' && 'text-yellow-600',
                variant === 'info' && 'text-primary-600'
              )}
              aria-hidden="true"
            />
          </div>
          <div className="flex-1">
            <h3
              id="confirm-dialog-title"
              className="text-lg font-semibold text-gray-900 mb-2"
            >
              {title}
            </h3>
            <p id="confirm-dialog-message" className="text-sm text-gray-600 mb-4">
              {message}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={onCancel}
                onKeyDown={(e) => handleKeyDown(e, onCancel)}
                className="btn btn-secondary"
                tabIndex={0}
                aria-label={cancelLabel}
              >
                {cancelLabel}
              </button>
              <button
                onClick={onConfirm}
                onKeyDown={(e) => handleKeyDown(e, onConfirm)}
                className={clsx('btn text-white', variantClasses[variant])}
                tabIndex={0}
                aria-label={confirmLabel}
              >
                {confirmLabel}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConfirmDialog

