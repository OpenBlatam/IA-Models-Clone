'use client'

import React from 'react'
import { FileText, User, Calendar, ExternalLink } from 'lucide-react'
import { Card, Badge } from '../ui'
import type { Paper } from '@/lib/api/types'
import { format } from '@/lib/utils'

interface PaperCardProps {
  paper: Paper
  onClick?: () => void
  className?: string
}

const PaperCard: React.FC<PaperCardProps> = ({
  paper,
  onClick,
  className,
}) => {
  return (
    <Card
      className={`cursor-pointer hover:shadow-md transition-all duration-200 ${className || ''}`}
      onClick={onClick}
    >
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <h3 className="font-semibold text-gray-900 line-clamp-2 flex-1">
            {paper.title || 'Untitled Paper'}
          </h3>
          <Badge variant="info" size="sm" className="ml-2 flex-shrink-0">
            {paper.source}
          </Badge>
        </div>

        {paper.authors && paper.authors.length > 0 && (
          <div className="flex items-center text-sm text-gray-600">
            <User className="w-4 h-4 mr-1 flex-shrink-0" />
            <span className="line-clamp-1">
              {paper.authors.join(', ')}
            </span>
          </div>
        )}

        {paper.abstract && (
          <p className="text-sm text-gray-600 line-clamp-3">
            {paper.abstract}
          </p>
        )}

        <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-200">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <FileText className="w-3 h-3" />
              {paper.sections_count} sections
            </span>
            <span>{format.bytes(paper.content_length)}</span>
          </div>
        </div>
      </div>
    </Card>
  )
}

export default PaperCard




