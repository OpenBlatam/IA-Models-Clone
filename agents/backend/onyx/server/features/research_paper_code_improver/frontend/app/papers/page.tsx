'use client'

import React, { useMemo, useState } from 'react'
import PageLayout from '@/components/layout/PageLayout'
import { Card, LoadingSpinner } from '@/components/ui'
import PaperUpload from '@/components/features/PaperUpload'
import PaperList from '@/components/features/PaperList'
import PaperSearch from '@/components/features/PaperSearch'
import { usePapers } from '@/hooks'
import { LoadingSpinner } from '@/components/ui'
import type { Paper } from '@/lib/api/types'

const PapersPage: React.FC = () => {
  const { data, isLoading, refetch } = usePapers(100)
  const [filteredPapers, setFilteredPapers] = useState<Paper[]>([])

  const handleUploadSuccess = () => {
    refetch()
  }

  const handleSearch = (papers: Paper[]) => {
    setFilteredPapers(papers)
  }

  const displayPapers = filteredPapers.length > 0 
    ? filteredPapers 
    : data?.papers || []

  return (
    <PageLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Research Papers</h1>
            <p className="mt-2 text-gray-600">
              Manage and browse your uploaded research papers
            </p>
          </div>
          <PaperUpload onUploadSuccess={handleUploadSuccess} />
        </div>

        {data && data.papers.length > 0 && (
          <PaperSearch
            papers={data.papers}
            onSearch={handleSearch}
          />
        )}

        <Card padding="none">
          <div className="px-6 py-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">
                All Papers
              </h2>
              <span className="text-sm text-gray-600">
                {displayPapers.length} of {data?.total || 0} papers
              </span>
            </div>
          </div>
          <div className="p-6">
            {isLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <PaperList papers={displayPapers} />
            )}
          </div>
        </Card>
      </div>
    </PageLayout>
  )
}

export default PapersPage

