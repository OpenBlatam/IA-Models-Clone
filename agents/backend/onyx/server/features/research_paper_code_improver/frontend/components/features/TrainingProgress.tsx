'use client'

import React from 'react'
import { Brain, CheckCircle, Clock, AlertCircle } from 'lucide-react'
import { Card, ProgressBar, Badge } from '../ui'

interface TrainingProgressProps {
  status: 'idle' | 'training' | 'completed' | 'error'
  progress?: number
  currentEpoch?: number
  totalEpochs?: number
  message?: string
}

const TrainingProgress: React.FC<TrainingProgressProps> = ({
  status,
  progress = 0,
  currentEpoch,
  totalEpochs,
  message,
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-600" />
      case 'training':
        return <Clock className="w-6 h-6 text-primary-600 animate-spin" />
      case 'error':
        return <AlertCircle className="w-6 h-6 text-red-600" />
      default:
        return <Brain className="w-6 h-6 text-gray-400" />
    }
  }

  const getStatusBadge = () => {
    switch (status) {
      case 'completed':
        return <Badge variant="success">Completed</Badge>
      case 'training':
        return <Badge variant="info">Training</Badge>
      case 'error':
        return <Badge variant="error">Error</Badge>
      default:
        return <Badge variant="default">Idle</Badge>
    }
  }

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h3 className="font-semibold text-gray-900">Training Status</h3>
              {currentEpoch && totalEpochs && (
                <p className="text-sm text-gray-600">
                  Epoch {currentEpoch} of {totalEpochs}
                </p>
              )}
            </div>
          </div>
          {getStatusBadge()}
        </div>

        {status === 'training' && (
          <ProgressBar
            value={progress}
            label="Training Progress"
            showPercentage
            variant="primary"
          />
        )}

        {message && (
          <div className="p-3 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">{message}</p>
          </div>
        )}

        {status === 'completed' && (
          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
            <p className="text-sm text-green-800 font-medium">
              Training completed successfully!
            </p>
          </div>
        )}

        {status === 'error' && (
          <div className="p-4 bg-red-50 rounded-lg border border-red-200">
            <p className="text-sm text-red-800 font-medium">
              Training failed. Please check the error message above.
            </p>
          </div>
        )}
      </div>
    </Card>
  )
}

export default TrainingProgress




