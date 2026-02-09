'use client'

import React, { useState } from 'react'
import { History, Clock, FileCode, ExternalLink } from 'lucide-react'
import { Card, Badge, Button, EmptyState } from '../ui'
import { Modal } from '../ui'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface CodeHistoryItem {
  id: string
  timestamp: string
  repo?: string
  filePath?: string
  originalCode: string
  improvedCode: string
  improvementsCount: number
  language?: string
}

interface CodeHistoryProps {
  items?: CodeHistoryItem[]
}

const CodeHistory: React.FC<CodeHistoryProps> = ({ items = [] }) => {
  const [selectedItem, setSelectedItem] = useState<CodeHistoryItem | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  // Mock data for demonstration - in real app, this would come from API/localStorage
  const historyItems: CodeHistoryItem[] = items.length > 0 ? items : []

  const handleViewDetails = (item: CodeHistoryItem) => {
    setSelectedItem(item)
    setIsModalOpen(true)
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  if (historyItems.length === 0) {
    return (
      <Card>
        <EmptyState
          icon={History}
          title="No History"
          description="Your code improvement history will appear here"
        />
      </Card>
    )
  }

  return (
    <>
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">
              Improvement History
            </h2>
            <Badge variant="info">{historyItems.length} items</Badge>
          </div>

          <div className="space-y-3">
            {historyItems.map((item) => (
              <div
                key={item.id}
                className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-sm transition-all cursor-pointer"
                onClick={() => handleViewDetails(item)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    {item.repo && item.filePath ? (
                      <div className="flex items-center gap-2 mb-2">
                        <FileCode className="w-4 h-4 text-gray-400" />
                        <span className="text-sm font-medium text-gray-900">
                          {item.repo}/{item.filePath}
                        </span>
                      </div>
                    ) : (
                      <div className="text-sm font-medium text-gray-900 mb-2">
                        Code Improvement
                      </div>
                    )}
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(item.timestamp)}
                      </div>
                      <Badge variant="success" size="sm">
                        {item.improvementsCount} improvements
                      </Badge>
                      {item.language && (
                        <Badge variant="default" size="sm">
                          {item.language}
                        </Badge>
                      )}
                    </div>
                  </div>
                  <Button variant="ghost" size="sm">
                    <ExternalLink className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {selectedItem && (
        <Modal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          title="Code Improvement Details"
          size="xl"
        >
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-2">Original Code</h3>
                <div className="overflow-x-auto">
                  <SyntaxHighlighter
                    language={selectedItem.language || 'python'}
                    style={vscDarkPlus}
                    customStyle={{ borderRadius: '8px', fontSize: '14px' }}
                  >
                    {selectedItem.originalCode}
                  </SyntaxHighlighter>
                </div>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Improved Code</h3>
                <div className="overflow-x-auto">
                  <SyntaxHighlighter
                    language={selectedItem.language || 'python'}
                    style={vscDarkPlus}
                    customStyle={{ borderRadius: '8px', fontSize: '14px' }}
                  >
                    {selectedItem.improvedCode}
                  </SyntaxHighlighter>
                </div>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </>
  )
}

export default CodeHistory




