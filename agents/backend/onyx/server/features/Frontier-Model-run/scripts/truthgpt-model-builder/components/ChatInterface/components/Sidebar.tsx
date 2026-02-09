/**
 * Sidebar Component
 * Sidebar with panels for history, settings, bookmarks, etc.
 */

'use client'

import React, { memo, useState } from 'react'
import { History, Bookmark, Settings, Archive, Tag, MessageSquare, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  activePanel?: 'history' | 'bookmarks' | 'settings' | 'archive' | 'tags'
  onPanelChange?: (panel: string | null) => void
  children?: React.ReactNode
}

export const Sidebar = memo(function Sidebar({
  isOpen,
  onClose,
  activePanel,
  onPanelChange,
  children,
}: SidebarProps) {
  const [currentPanel, setCurrentPanel] = useState<string | null>(activePanel || null)

  const handlePanelChange = (panel: string) => {
    const newPanel = currentPanel === panel ? null : panel
    setCurrentPanel(newPanel)
    onPanelChange?.(newPanel)
  }

  const panels = [
    { id: 'history', icon: History, label: 'Historial' },
    { id: 'bookmarks', icon: Bookmark, label: 'Marcadores' },
    { id: 'tags', icon: Tag, label: 'Etiquetas' },
    { id: 'archive', icon: Archive, label: 'Archivo' },
    { id: 'settings', icon: Settings, label: 'Configuración' },
  ]

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="sidebar__backdrop"
          />

          {/* Sidebar */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="sidebar"
          >
            <div className="sidebar__header">
              <h2 className="sidebar__title">Panel</h2>
              <button
                type="button"
                onClick={onClose}
                className="sidebar__close"
                aria-label="Cerrar panel"
              >
                <X size={20} />
              </button>
            </div>

            <div className="sidebar__navigation">
              {panels.map(panel => (
                <button
                  key={panel.id}
                  type="button"
                  onClick={() => handlePanelChange(panel.id)}
                  className={`sidebar__nav-item ${
                    currentPanel === panel.id ? 'sidebar__nav-item--active' : ''
                  }`}
                >
                  <panel.icon size={20} />
                  <span>{panel.label}</span>
                </button>
              ))}
            </div>

            <div className="sidebar__content">
              {children || (
                <div className="sidebar__empty">
                  <MessageSquare size={48} />
                  <p>Selecciona un panel</p>
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
})

export default Sidebar




