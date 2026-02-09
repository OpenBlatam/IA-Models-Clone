'use client'

import React, { useState } from 'react'
import { FileText } from 'lucide-react'
import { Card, LoadingSpinner } from '../ui'
import PaperDetail from './PaperDetail'
import PaperCard from './PaperCard'
import DataTable from './DataTable'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui'
import type { Paper } from '@/lib/api/types'
import { format } from '@/lib/utils'

interface PaperListProps {
  papers: Paper[]
  isLoading?: boolean
  onPaperClick?: (paper: Paper) => void
}

const PaperList: React.FC<PaperListProps> = ({
  papers,
  isLoading,
  onPaperClick,
}) => {
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)
  const [isDetailOpen, setIsDetailOpen] = useState(false)
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid')

  const handlePaperClick = (paper: Paper) => {
    setSelectedPaper(paper)
    setIsDetailOpen(true)
    onPaperClick?.(paper)
  }

  const tableColumns = [
    {
      key: 'title',
      header: 'Title',
      render: (value: string, row: Paper) => (
        <span className="font-medium text-gray-900">{value || 'Untitled'}</span>
      ),
      sortable: true,
    },
    {
      key: 'authors',
      header: 'Authors',
      render: (value: string[]) =>
        value && value.length > 0 ? (
          <span className="text-sm text-gray-600">
            {value.slice(0, 2).join(', ')}
            {value.length > 2 && ` +${value.length - 2}`}
          </span>
        ) : (
          <span className="text-gray-400">-</span>
        ),
    },
    {
      key: 'source',
      header: 'Source',
      render: (value: string) => (
        <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
          {value}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'sections_count',
      header: 'Sections',
      render: (value: number) => (
        <span className="text-sm text-gray-600">{value}</span>
      ),
      sortable: true,
    },
    {
      key: 'content_length',
      header: 'Size',
      render: (value: number) => (
        <span className="text-sm text-gray-600">{format.bytes(value)}</span>
      ),
      sortable: true,
    },
  ]

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (papers.length === 0) {
    return (
      <Card>
        <div className="text-center py-12">
          <FileText className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-600 text-lg">No papers found</p>
          <p className="text-gray-500 text-sm mt-2">
            Upload your first research paper to get started
          </p>
        </div>
      </Card>
    )
  }

  return (
    <>
      <Card>
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'grid' | 'table')}>
          <div className="flex items-center justify-between mb-4">
            <TabsList>
              <TabsTrigger value="grid">Grid View</TabsTrigger>
              <TabsTrigger value="table">Table View</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="grid">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {papers.map((paper) => (
                <PaperCard
                  key={paper.id}
                  paper={paper}
                  onClick={() => handlePaperClick(paper)}
                />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="table">
            <DataTable
              data={papers}
              columns={tableColumns}
              pageSize={10}
              searchable
              searchPlaceholder="Search papers..."
              onRowClick={(row) => handlePaperClick(row as Paper)}
            />
          </TabsContent>
        </Tabs>
      </Card>

      <PaperDetail
        paper={selectedPaper}
        isOpen={isDetailOpen}
        onClose={() => setIsDetailOpen(false)}
      />
    </>
  )
}

export default PaperList
