/**
 * Message Optimizer Component
 * Optimizes message rendering with virtualization
 */

'use client'

import { useMemo, useRef, useEffect, useState } from 'react'
import { useVirtualScroll } from '@/lib/optimization-utils'
import Message from './Message'

interface MessageOptimizerProps {
  messages: any[]
  containerHeight?: number
  itemHeight?: number
}

export default function MessageOptimizer({
  messages,
  containerHeight = 600,
  itemHeight = 100
}: MessageOptimizerProps) {
  const [containerRef, setContainerRef] = useState<HTMLDivElement | null>(null)
  const [actualHeight, setActualHeight] = useState(containerHeight)

  useEffect(() => {
    if (containerRef) {
      const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
          setActualHeight(entry.contentRect.height)
        }
      })

      resizeObserver.observe(containerRef)

      return () => {
        resizeObserver.disconnect()
      }
    }
  }, [containerRef])

  const { visibleItems, totalHeight, offsetY } = useVirtualScroll(
    messages,
    actualHeight,
    itemHeight
  )

  return (
    <div
      ref={setContainerRef}
      className="relative h-full overflow-hidden"
      style={{ height: containerHeight }}
    >
      <div
        style={{
          height: totalHeight,
          transform: `translateY(${offsetY}px)`,
          willChange: 'transform'
        }}
        className="absolute inset-0"
      >
        {visibleItems.map((message, index) => (
          <div
            key={message.id || index}
            style={{
              height: itemHeight,
              position: 'absolute',
              top: (messages.indexOf(message) * itemHeight) - offsetY,
              width: '100%'
            }}
          >
            <Message message={message} />
          </div>
        ))}
      </div>
    </div>
  )
}


