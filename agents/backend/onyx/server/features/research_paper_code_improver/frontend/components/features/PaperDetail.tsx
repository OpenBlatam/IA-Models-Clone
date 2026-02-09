'use client'

import React from 'react'
import { X, FileText, User, Calendar, ExternalLink } from 'lucide-react'
import { Modal, Badge, Card, Button } from '../ui'
import type { Paper } from '@/lib/api/types'

interface PaperDetailProps {
  paper: Paper | null
  isOpen: boolean
  onClose: () => void
}

const PaperDetail: React.FC<PaperDetailProps> = ({ paper, isOpen, onClose }) => {
  if (!paper) return null

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={paper.title} size="xl">
      <div className="space-y-6">
        {/* Metadata */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
              <FileText className="w-4 h-4" />
              <span>Source</span>
            </div>
            <Badge variant="info">{paper.source}</Badge>
          </div>
          <div>
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
              <FileText className="w-4 h-4" />
              <span>Sections</span>
            </div>
            <p className="text-sm font-medium text-gray-900">
              {paper.sections_count}
            </p>
          </div>
        </div>

        {/* Authors */}
        {paper.authors && paper.authors.length > 0 && (
          <div>
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
              <User className="w-4 h-4" />
              <span>Authors</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {paper.authors.map((author, index) => (
                <Badge key={index} variant="default" size="sm">
                  {author}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Abstract */}
        {paper.abstract && (
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-2">
              Abstract
            </h3>
            <p className="text-sm text-gray-700 leading-relaxed">
              {paper.abstract}
            </p>
          </div>
        )}

        {/* Statistics */}
        <Card padding="sm">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-primary-600">
                {paper.sections_count}
              </p>
              <p className="text-xs text-gray-600 mt-1">Sections</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary-600">
                {(paper.content_length / 1024).toFixed(1)} KB
              </p>
              <p className="text-xs text-gray-600 mt-1">Content Size</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary-600">
                {paper.authors?.length || 0}
              </p>
              <p className="text-xs text-gray-600 mt-1">Authors</p>
            </div>
          </div>
        </Card>

        {/* Metadata */}
        {paper.metadata && Object.keys(paper.metadata).length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-2">
              Additional Metadata
            </h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <pre className="text-xs text-gray-700 overflow-x-auto">
                {JSON.stringify(paper.metadata, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </Modal>
  )
}

export default PaperDetail




