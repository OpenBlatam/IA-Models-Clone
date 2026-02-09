/**
 * Modals Component
 * Collection of reusable modal components
 */

'use client'

import React, { memo } from 'react'
import { X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface BaseModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  showCloseButton?: boolean
}

export const BaseModal = memo(function BaseModal({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  showCloseButton = true,
}: BaseModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="modal__backdrop"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className={`modal modal--${size}`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal__header">
              <h2 className="modal__title">{title}</h2>
              {showCloseButton && (
                <button
                  type="button"
                  onClick={onClose}
                  className="modal__close"
                  aria-label="Cerrar"
                >
                  <X size={20} />
                </button>
              )}
            </div>
            <div className="modal__content">{children}</div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
})

interface ConfirmDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  variant?: 'danger' | 'warning' | 'info'
}

export const ConfirmDialog = memo(function ConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirmar',
  cancelText = 'Cancelar',
  variant = 'info',
}: ConfirmDialogProps) {
  const handleConfirm = () => {
    onConfirm()
    onClose()
  }

  return (
    <BaseModal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size="sm"
    >
      <div className="confirm-dialog">
        <p className="confirm-dialog__message">{message}</p>
        <div className="confirm-dialog__actions">
          <button
            type="button"
            onClick={onClose}
            className="confirm-dialog__button confirm-dialog__button--cancel"
          >
            {cancelText}
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className={`confirm-dialog__button confirm-dialog__button--confirm confirm-dialog__button--${variant}`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </BaseModal>
  )
})

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  children: React.ReactNode
}

export const SettingsModal = memo(function SettingsModal({
  isOpen,
  onClose,
  children,
}: SettingsModalProps) {
  return (
    <BaseModal
      isOpen={isOpen}
      onClose={onClose}
      title="Configuración"
      size="lg"
    >
      <div className="settings-modal">
        {children}
      </div>
    </BaseModal>
  )
})

interface ExportModalProps {
  isOpen: boolean
  onClose: () => void
  onExport: (format: string) => void
  availableFormats?: string[]
}

export const ExportModal = memo(function ExportModal({
  isOpen,
  onClose,
  onExport,
  availableFormats = ['json', 'txt', 'md', 'html', 'csv'],
}: ExportModalProps) {
  return (
    <BaseModal
      isOpen={isOpen}
      onClose={onClose}
      title="Exportar Mensajes"
      size="md"
    >
      <div className="export-modal">
        <p className="export-modal__description">
          Selecciona el formato para exportar tus mensajes:
        </p>
        <div className="export-modal__formats">
          {availableFormats.map(format => (
            <button
              key={format}
              type="button"
              onClick={() => {
                onExport(format)
                onClose()
              }}
              className="export-modal__format-button"
            >
              {format.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
    </BaseModal>
  )
})

export default {
  BaseModal,
  ConfirmDialog,
  SettingsModal,
  ExportModal,
}




