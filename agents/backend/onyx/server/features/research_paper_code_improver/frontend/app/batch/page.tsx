'use client'

import React from 'react'
import PageLayout from '@/components/layout/PageLayout'
import BatchProcessor from '@/components/features/BatchProcessor'

const BatchPage: React.FC = () => {
  return (
    <PageLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Batch Processing</h1>
          <p className="mt-2 text-gray-600">
            Process multiple code files at once for efficient improvements
          </p>
        </div>

        <BatchProcessor />
      </div>
    </PageLayout>
  )
}

export default BatchPage



