/**
 * Hook para Manejar Modales
 * =========================
 * 
 * Hook para gestionar el estado de modales
 */

import { useState, useCallback } from 'react'

export interface UseModalResult {
  isOpen: boolean
  open: () => void
  close: () => void
  toggle: () => void
}

/**
 * Hook para manejar modales
 */
export function useModal(initialState: boolean = false): UseModalResult {
  const [isOpen, setIsOpen] = useState(initialState)

  const open = useCallback(() => {
    setIsOpen(true)
  }, [])

  const close = useCallback(() => {
    setIsOpen(false)
  }, [])

  const toggle = useCallback(() => {
    setIsOpen(prev => !prev)
  }, [])

  return {
    isOpen,
    open,
    close,
    toggle
  }
}







