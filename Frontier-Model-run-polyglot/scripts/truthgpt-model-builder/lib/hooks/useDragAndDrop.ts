/**
 * Hook useDragAndDrop
 * ===================
 * 
 * Hook para manejo de drag and drop
 */

import { useRef, useEffect, useCallback } from 'react'
import { setupDragDrop, DragDropConfig } from '../utils/dragDropUtils'

export interface UseDragAndDropReturn {
  dragRef: React.RefObject<HTMLElement>
  dropRef: React.RefObject<HTMLElement>
  isDragging: boolean
}

/**
 * Hook para drag and drop
 */
export function useDragAndDrop(
  config: {
    dragConfig?: DragDropConfig
    dropConfig?: DragDropConfig
  } = {}
): UseDragAndDropReturn {
  const dragRef = useRef<HTMLElement>(null)
  const dropRef = useRef<HTMLElement>(null)
  const isDraggingRef = useRef(false)

  useEffect(() => {
    const cleanupFunctions: (() => void)[] = []

    if (dragRef.current && config.dragConfig) {
      const cleanup = setupDragDrop(dragRef.current, {
        ...config.dragConfig,
        onDragStart: (e, data) => {
          isDraggingRef.current = true
          config.dragConfig?.onDragStart?.(e, data)
        },
        onDragEnd: (e) => {
          isDraggingRef.current = false
          config.dragConfig?.onDragEnd?.(e)
        }
      })
      cleanupFunctions.push(cleanup)
    }

    if (dropRef.current && config.dropConfig) {
      const cleanup = setupDragDrop(dropRef.current, config.dropConfig)
      cleanupFunctions.push(cleanup)
    }

    return () => {
      cleanupFunctions.forEach(cleanup => cleanup())
    }
  }, [config.dragConfig, config.dropConfig])

  return {
    dragRef,
    dropRef,
    isDragging: isDraggingRef.current
  }
}






