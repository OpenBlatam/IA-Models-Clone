'use client'

import React from 'react'
import PageLayout from '@/components/layout/PageLayout'
import RepositoryAnalyzer from '@/components/features/RepositoryAnalyzer'

const AnalyzePage: React.FC = () => {
  return (
    <PageLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Repository Analyzer</h1>
          <p className="mt-2 text-gray-600">
            Analyze entire GitHub repositories and get improvement suggestions
          </p>
        </div>

        <RepositoryAnalyzer />
      </div>
    </PageLayout>
  )
}

export default AnalyzePage



