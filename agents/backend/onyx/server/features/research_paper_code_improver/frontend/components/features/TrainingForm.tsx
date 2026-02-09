'use client'

import React, { useState } from 'react'
import { Brain, Play } from 'lucide-react'
import { Card, Button, Input, Badge, LoadingSpinner } from '../ui'
import TrainingProgress from './TrainingProgress'
import StatusIndicator from './StatusIndicator'
import ModelSelector from './ModelSelector'
import { apiClient } from '@/lib/api'
import toast from 'react-hot-toast'
import type { Paper } from '@/lib/api/types'

interface TrainingFormProps {
  papers: Paper[]
  onTrainingComplete?: () => void
}

const TrainingForm: React.FC<TrainingFormProps> = ({
  papers,
  onTrainingComplete,
}) => {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'training' | 'completed' | 'error'>('idle')
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [epochs, setEpochs] = useState(3)
  const [modelName, setModelName] = useState('gpt-4')
  const [useAllPapers, setUseAllPapers] = useState(true)
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([])
  const [showModelSelector, setShowModelSelector] = useState(false)

  const handleStartTraining = async () => {
    if (papers.length === 0) {
      toast.error('No papers available for training')
      return
    }

    setIsTraining(true)
    setTrainingStatus('training')
    setTrainingProgress(0)
    setCurrentEpoch(0)

    try {
      // Simulate progress updates (in real app, this would come from WebSocket or polling)
      const progressInterval = setInterval(() => {
        setTrainingProgress((prev) => {
          if (prev >= 100) {
            clearInterval(progressInterval)
            return 100
          }
          return prev + 10
        })
        setCurrentEpoch((prev) => {
          if (prev < epochs) {
            return prev + 1
          }
          return prev
        })
      }, 1000)

      const response = await apiClient.trainModel({
        paper_ids: useAllPapers ? undefined : selectedPaperIds,
        model_name: modelName,
        epochs,
        use_all_papers: useAllPapers,
      })

      clearInterval(progressInterval)
      setTrainingProgress(100)
      setTrainingStatus('completed')
      toast.success(
        `Training completed! Model ID: ${response.model_id}. Status: ${response.status}`
      )
      onTrainingComplete?.()
    } catch (error) {
      setTrainingStatus('error')
      toast.error(
        error instanceof Error ? error.message : 'Failed to start training'
      )
    } finally {
      setIsTraining(false)
    }
  }

  const togglePaperSelection = (paperId: string) => {
    if (selectedPaperIds.includes(paperId)) {
      setSelectedPaperIds(selectedPaperIds.filter((id) => id !== paperId))
    } else {
      setSelectedPaperIds([...selectedPaperIds, paperId])
    }
  }

  return (
    <Card>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-primary-600" />
            <h2 className="text-xl font-semibold">Train Model</h2>
          </div>
          <StatusIndicator
            status={
              trainingStatus === 'completed'
                ? 'success'
                : trainingStatus === 'error'
                ? 'error'
                : trainingStatus === 'training'
                ? 'pending'
                : 'warning'
            }
            label={
              trainingStatus === 'completed'
                ? 'Completed'
                : trainingStatus === 'error'
                ? 'Error'
                : trainingStatus === 'training'
                ? 'Training'
                : 'Ready'
            }
          />
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model Name
            </label>
            <div className="flex gap-2">
              <Input
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="gpt-4"
                disabled={isTraining}
                className="flex-1"
              />
              <div className="pt-6">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowModelSelector(!showModelSelector)}
                  disabled={isTraining}
                  title="Select Model"
                >
                  <Brain className="w-4 h-4" />
                </Button>
              </div>
            </div>
            {showModelSelector && (
              <div className="mt-2">
                <ModelSelector
                  selectedModelId={modelName}
                  onSelect={(id) => {
                    setModelName(id || 'gpt-4')
                    setShowModelSelector(false)
                  }}
                  showDefault={false}
                />
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Epochs
            </label>
            <Input
              type="number"
              min="1"
              max="10"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 3)}
              disabled={isTraining}
            />
          </div>

          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useAllPapers}
                onChange={(e) => {
                  setUseAllPapers(e.target.checked)
                  if (e.target.checked) {
                    setSelectedPaperIds([])
                  }
                }}
                disabled={isTraining}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm font-medium text-gray-700">
                Use all papers ({papers.length} available)
              </span>
            </label>
          </div>

          {!useAllPapers && papers.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Papers
              </label>
              <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-2 space-y-2">
                {papers.map((paper) => (
                  <label
                    key={paper.id}
                    className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-2 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={selectedPaperIds.includes(paper.id)}
                      onChange={() => togglePaperSelection(paper.id)}
                      disabled={isTraining}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-sm text-gray-700 line-clamp-1">
                      {paper.title || 'Untitled'}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          <Button
            onClick={handleStartTraining}
            isLoading={isTraining}
            disabled={papers.length === 0 || (!useAllPapers && selectedPaperIds.length === 0)}
            className="w-full"
          >
            <Play className="w-4 h-4 mr-2" />
            Start Training
          </Button>
        </div>
      </div>

      {(isTraining || trainingStatus !== 'idle') && (
        <TrainingProgress
          status={trainingStatus}
          progress={trainingProgress}
          currentEpoch={currentEpoch}
          totalEpochs={epochs}
          message={
            trainingStatus === 'training'
              ? 'Training model with selected papers...'
              : trainingStatus === 'completed'
              ? 'Model training completed successfully!'
              : undefined
          }
        />
      )}
    </Card>
  )
}

export default TrainingForm
