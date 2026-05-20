/**
 * Rate Limit Indicator Component
 * Shows current rate limit status
 */

'use client'

import { useEffect, useState } from 'react'
import { AlertCircle, Clock, CheckCircle } from 'lucide-react'
import { motion } from 'framer-motion'
import { rateLimiter } from '@/lib/rate-limiter'

interface RateLimitIndicatorProps {
  className?: string
}

export default function RateLimitIndicator({ className = '' }: RateLimitIndicatorProps) {
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {
    const updateStatus = () => {
      try {
        const currentStatus = rateLimiter.getStatus()
        setStatus(currentStatus)
      } catch (error) {
        console.error('Error fetching rate limit status:', error)
      }
    }

    updateStatus()
    const interval = setInterval(updateStatus, 1000) // Update every second

    return () => clearInterval(interval)
  }, [])

  if (!status) return null

  const isLimited = !status.allowed
  const waitSeconds = status.waitTimeMs ? Math.ceil(status.waitTimeMs / 1000) : 0

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
        isLimited
          ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
          : 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
      } ${className}`}
    >
      {isLimited ? (
        <>
          <AlertCircle className="w-4 h-4 text-red-500" />
          <span className="text-sm text-red-700 dark:text-red-300">
            Rate limit: Wait {waitSeconds}s
          </span>
        </>
      ) : (
        <>
          <CheckCircle className="w-4 h-4 text-green-500" />
          <div className="flex items-center gap-2">
            <span className="text-sm text-green-700 dark:text-green-300">
              {status.tokensAvailable.toFixed(1)} tokens available
            </span>
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Clock className="w-3 h-3" />
              <span>{status.tokensPerSecond}/s</span>
            </div>
          </div>
        </>
      )}
    </motion.div>
  )
}


