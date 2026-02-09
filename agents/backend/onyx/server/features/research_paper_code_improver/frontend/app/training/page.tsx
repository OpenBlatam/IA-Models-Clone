'use client'

import React from 'react'
import { useQuery } from '@tanstack/react-query'
import PageLayout from '@/components/layout/PageLayout'
import TrainingForm from '@/components/features/TrainingForm'
import { Card, LoadingSpinner } from '@/components/ui'
import { apiClient } from '@/lib/api'

const TrainingPage: React.FC = () => {
  const { data: papers, isLoading, refetch } = useQuery({
    queryKey: ['papers'],
    queryFn: () => apiClient.listPapers(100),
  })

  const handleTrainingComplete = () => {
    refetch()
  }

  return (
    <PageLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>
          <p className="mt-2 text-gray-600">
            Train AI models using your research papers
          </p>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12">
            <LoadingSpinner size="lg" />
          </div>
        ) : (
          <TrainingForm
            papers={papers?.papers || []}
            onTrainingComplete={handleTrainingComplete}
          />
        )}

        {papers && papers.papers.length === 0 && (
          <Card>
            <div className="text-center py-8">
              <p className="text-gray-600">
                No papers available. Upload papers first to start training.
              </p>
            </div>
          </Card>
        )}
      </div>
    </PageLayout>
  )
}

export default TrainingPage




