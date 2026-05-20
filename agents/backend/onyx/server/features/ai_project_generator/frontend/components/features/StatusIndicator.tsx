'use client'

import { useCallback } from 'react'
import clsx from 'clsx'
import { Play, Square, Wifi, WifiOff } from 'lucide-react'
import Button from '@/components/ui/Button'

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
  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        {isConnected ? (
          <Wifi className="w-5 h-5 text-green-500" aria-hidden="true" />
        ) : (
          <WifiOff className="w-5 h-5 text-red-500" aria-hidden="true" />
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
          aria-hidden="true"
        />
        <span className="text-sm text-gray-600">{isRunning ? 'Running' : 'Stopped'}</span>
      </div>

      <span className="text-sm text-gray-600">Queue: {queueSize}</span>

      {isRunning ? (
        <Button
          variant="danger"
          size="sm"
          leftIcon={<Square className="w-4 h-4" />}
          onClick={onStop}
          aria-label="Stop Generator"
        >
          Stop
        </Button>
      ) : (
        <Button
          variant="primary"
          size="sm"
          leftIcon={<Play className="w-4 h-4" />}
          onClick={onStart}
          aria-label="Start Generator"
        >
          Start
        </Button>
      )}
    </div>
  )
}

export default StatusIndicator

