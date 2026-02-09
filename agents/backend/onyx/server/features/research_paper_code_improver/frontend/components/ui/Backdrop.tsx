'use client'

import React from 'react'
import { clsx } from 'clsx'

interface BackdropProps {
  isOpen: boolean
  onClose?: () => void
  className?: string
  children?: React.ReactNode
}

const Backdrop: React.FC<BackdropProps> = ({
  isOpen,
  onClose,
  className,
  children,
}) => {
  if (!isOpen) return null

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget && onClose) {
      onClose()
    }
  }

  return (
    <div
      className={clsx(
        'fixed inset-0 z-40 bg-black bg-opacity-50 animate-fade-in',
        className
      )}
      onClick={handleClick}
      role="presentation"
    >
      {children}
    </div>
  )
}

export default Backdrop




