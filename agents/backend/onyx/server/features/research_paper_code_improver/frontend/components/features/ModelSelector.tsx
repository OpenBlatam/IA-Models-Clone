'use client'

import React, { useState, useEffect } from 'react'
import { Brain, Check } from 'lucide-react'
import { Card, Button, Badge } from '../ui'
import { apiClient } from '@/lib/api'
import { useQuery } from '@tanstack/react-query'

interface ModelSelectorProps {
  selectedModelId?: string
  onSelect?: (modelId: string | null) => void
  showDefault?: boolean
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModelId,
  onSelect,
  showDefault = true,
}) => {
  const [availableModels, setAvailableModels] = useState<
    Array<{ id: string; name: string; status: string }>
  >([])

  // In a real app, this would fetch from API
  // For now, we'll use a mock list
  const models = [
    { id: 'default', name: 'Default Model', status: 'ready' },
    { id: 'gpt-4', name: 'GPT-4', status: 'ready' },
    { id: 'claude-3', name: 'Claude 3', status: 'ready' },
  ]

  const handleSelect = (modelId: string | null) => {
    onSelect?.(modelId)
  }

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary-600" />
          <h3 className="font-semibold text-gray-900">Select Model</h3>
        </div>

        <div className="space-y-2">
          {showDefault && (
            <button
              onClick={() => handleSelect(null)}
              className={`w-full flex items-center justify-between p-3 rounded-lg border-2 transition-colors ${
                selectedModelId === null || !selectedModelId
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-primary-300'
              }`}
            >
              <div className="flex items-center gap-3">
                <Brain className="w-5 h-5 text-primary-600" />
                <div className="text-left">
                  <p className="font-medium text-gray-900">Default Model</p>
                  <p className="text-xs text-gray-500">System default</p>
                </div>
              </div>
              {(selectedModelId === null || !selectedModelId) && (
                <Check className="w-5 h-5 text-primary-600" />
              )}
            </button>
          )}

          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => handleSelect(model.id)}
              className={`w-full flex items-center justify-between p-3 rounded-lg border-2 transition-colors ${
                selectedModelId === model.id
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-primary-300'
              }`}
            >
              <div className="flex items-center gap-3">
                <Brain className="w-5 h-5 text-primary-600" />
                <div className="text-left">
                  <p className="font-medium text-gray-900">{model.name}</p>
                  <p className="text-xs text-gray-500">Status: {model.status}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge
                  variant={model.status === 'ready' ? 'success' : 'warning'}
                  size="sm"
                >
                  {model.status}
                </Badge>
                {selectedModelId === model.id && (
                  <Check className="w-5 h-5 text-primary-600" />
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </Card>
  )
}

export default ModelSelector



