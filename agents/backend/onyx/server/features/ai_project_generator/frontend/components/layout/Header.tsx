'use client'

import StatusIndicator from '@/components/features/StatusIndicator'

interface HeaderProps {
  status: {
    isRunning: boolean
    isConnected: boolean
    queueSize: number
  }
  onStart: () => void
  onStop: () => void
}

const Header = ({ status, onStart, onStop }: HeaderProps) => {
  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">AI Project Generator</h1>
          <StatusIndicator
            isRunning={status.isRunning}
            isConnected={status.isConnected}
            queueSize={status.queueSize}
            onStart={onStart}
            onStop={onStop}
          />
        </div>
      </div>
    </header>
  )
}

export default Header

