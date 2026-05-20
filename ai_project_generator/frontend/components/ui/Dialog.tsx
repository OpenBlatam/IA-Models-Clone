'use client'

import { ReactNode } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import Modal from './Modal'
import Button from './Button'

interface DialogProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  children: ReactNode
  footer?: ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showCloseButton?: boolean
  className?: string
}

const Dialog = ({
  isOpen,
  onClose,
  title,
  children,
  footer,
  size = 'md',
  showCloseButton = true,
  className,
}: DialogProps) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} size={size}>
      <div className={cn('flex flex-col', className)}>
        {title && (
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
            {showCloseButton && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onClose}
                leftIcon={<X className="w-4 h-4" />}
              />
            )}
          </div>
        )}
        <div className="p-6 flex-1 overflow-y-auto">{children}</div>
        {footer && (
          <div className="p-6 border-t border-gray-200 flex items-center justify-end gap-3">
            {footer}
          </div>
        )}
      </div>
    </Modal>
  )
}

export default Dialog

