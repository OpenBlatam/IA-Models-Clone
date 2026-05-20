'use client'

import { ReactNode } from 'react'
import { AlertCircle, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import Modal from './Modal'
import Button from './Button'

interface AlertDialogProps {
  isOpen: boolean
  onClose: () => void
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel?: () => void
  variant?: 'default' | 'danger' | 'warning'
  icon?: ReactNode
}

const AlertDialog = ({
  isOpen,
  onClose,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  variant = 'default',
  icon,
}: AlertDialogProps) => {
  const handleConfirm = () => {
    onConfirm()
    onClose()
  }

  const handleCancel = () => {
    onCancel?.()
    onClose()
  }

  const variantClasses = {
    default: 'text-blue-600 bg-blue-50',
    danger: 'text-red-600 bg-red-50',
    warning: 'text-yellow-600 bg-yellow-50',
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="sm">
      <div className="p-6">
        <div className="flex items-start gap-4 mb-4">
          {icon || (
            <div className={cn('p-2 rounded-full', variantClasses[variant])}>
              <AlertCircle className="w-6 h-6" />
            </div>
          )}
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
            <p className="text-gray-600">{message}</p>
          </div>
        </div>
        <div className="flex items-center justify-end gap-3">
          {onCancel && (
            <Button variant="secondary" onClick={handleCancel}>
              {cancelLabel}
            </Button>
          )}
          <Button
            variant={variant === 'danger' ? 'danger' : 'primary'}
            onClick={handleConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  )
}

export default AlertDialog

