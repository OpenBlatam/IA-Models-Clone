'use client'

import { useCallback } from 'react'
import clsx from 'clsx'
import { Play, Square, Wifi, WifiOff } from 'lucide-react'

interface StatusIndicatorProps {
  isRunning: boolean
  isConnected: boolean
  queueSize: number
  onStart: () => void
  onStop: () => void
}

const StatusIndicator = ({
  isRunning,
  isConnected,
  queueSize,
  onStart,
  onStop,
}: StatusIndicatorProps) => {
  const handleKeyDown = useCallback((e: React.KeyboardEvent, action: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action()
    }
  }, [])

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        {isConnected ? (
          <Wifi className="w-5 h-5 text-green-500" />
        ) : (
          <WifiOff className="w-5 h-5 text-red-500" />
        )}
        <span className="text-sm text-gray-600">
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <div
          className={clsx(
            'w-3 h-3 rounded-full',
            isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
          )}
        />
        <span className="text-sm text-gray-600">
          {isRunning ? 'Running' : 'Stopped'}
        </span>
      </div>

      <span className="text-sm text-gray-600">Queue: {queueSize}</span>

      {isRunning ? (
        <button
          onClick={onStop}
          className="btn btn-danger flex items-center gap-2"
          tabIndex={0}
          aria-label="Stop Generator"
          onKeyDown={(e) => handleKeyDown(e, onStop)}
        >
          <Square className="w-4 h-4" />
          Stop
        </button>
      ) : (
        <button
          onClick={onStart}
          className="btn btn-primary flex items-center gap-2"
          tabIndex={0}
          aria-label="Start Generator"
          onKeyDown={(e) => handleKeyDown(e, onStart)}
        >
          <Play className="w-4 h-4" />
          Start
        </button>
      )}
    </div>
  )
}

export default StatusIndicator

