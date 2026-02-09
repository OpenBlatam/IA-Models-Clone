'use client'

import { useEffect, useState, ReactNode } from 'react'
import { createPortal } from 'react-dom'

interface PortalProps {
  children: ReactNode
  container?: HTMLElement | null
  className?: string
}

const Portal = ({ children, container, className }: PortalProps) => {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  const targetContainer = container || (typeof document !== 'undefined' ? document.body : null)

  if (!targetContainer) {
    return null
  }

  const content = className ? <div className={className}>{children}</div> : children

  return createPortal(content, targetContainer)
}

export default Portal

