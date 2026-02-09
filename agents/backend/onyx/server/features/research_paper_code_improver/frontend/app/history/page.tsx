'use client'

import React, { useEffect, useState } from 'react'
import PageLayout from '@/components/layout/PageLayout'
import CodeHistory from '@/components/features/CodeHistory'
import DataTable from '@/components/features/DataTable'
import { Card, Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui'
import { storage, storageKeys, format } from '@/lib/utils'

interface HistoryItem {
  id: string
  timestamp: string
  repo?: string
  filePath?: string
  originalCode: string
  improvedCode: string
  improvementsCount: number
  language?: string
}

const HistoryPage: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards')

  useEffect(() => {
    const items = storage.get<HistoryItem[]>(storageKeys.CODE_HISTORY, [])
    setHistoryItems(items || [])
  }, [])

  const tableColumns = [
    {
      key: 'timestamp',
      header: 'Date',
      render: (value: string) => format.dateTime(value),
      sortable: true,
    },
    {
      key: 'repo',
      header: 'Repository',
      render: (value: string, row: HistoryItem) =>
        value ? (
          <span className="font-mono text-sm">{value}</span>
        ) : (
          <span className="text-gray-400">Direct input</span>
        ),
      sortable: true,
    },
    {
      key: 'filePath',
      header: 'File Path',
      render: (value: string) =>
        value ? (
          <span className="font-mono text-xs">{value}</span>
        ) : (
          <span className="text-gray-400">-</span>
        ),
    },
    {
      key: 'language',
      header: 'Language',
      render: (value: string) =>
        value ? (
          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
            {value}
          </span>
        ) : (
          <span className="text-gray-400">-</span>
        ),
      sortable: true,
    },
    {
      key: 'improvementsCount',
      header: 'Improvements',
      render: (value: number) => (
        <span className="font-semibold text-green-600">{value}</span>
      ),
      sortable: true,
    },
  ]

  return (
    <PageLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Code Improvement History
            </h1>
            <p className="mt-2 text-gray-600">
              View and manage your code improvement history
            </p>
          </div>
        </div>

        <Card>
          <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'cards' | 'table')}>
            <TabsList>
              <TabsTrigger value="cards">Card View</TabsTrigger>
              <TabsTrigger value="table">Table View</TabsTrigger>
            </TabsList>

            <TabsContent value="cards">
              <CodeHistory items={historyItems} />
            </TabsContent>

            <TabsContent value="table">
              <div className="mt-4">
                <DataTable
                  data={historyItems}
                  columns={tableColumns}
                  pageSize={10}
                  searchable
                  searchPlaceholder="Search history..."
                />
              </div>
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </PageLayout>
  )
}

export default HistoryPage
