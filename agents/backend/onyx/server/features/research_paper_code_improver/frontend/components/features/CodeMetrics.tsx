'use client'

import React from 'react'
import { Code, FileText, AlertCircle, CheckCircle } from 'lucide-react'
import { Card, Badge } from '../ui'

interface CodeMetricsProps {
  original: string
  improved: string
  suggestions: Array<{
    type: string
    severity?: string
  }>
}

const CodeMetrics: React.FC<CodeMetricsProps> = ({
  original,
  improved,
  suggestions,
}) => {
  const metrics = {
    originalLines: original.split('\n').length,
    improvedLines: improved.split('\n').length,
    originalChars: original.length,
    improvedChars: improved.length,
    suggestionsCount: suggestions.length,
    highSeverity: suggestions.filter((s) => s.severity === 'high').length,
    mediumSeverity: suggestions.filter((s) => s.severity === 'medium').length,
    lowSeverity: suggestions.filter((s) => s.severity === 'low').length,
  }

  const lineDiff = metrics.improvedLines - metrics.originalLines
  const charDiff = metrics.improvedChars - metrics.originalChars

  return (
    <Card>
      <h3 className="font-semibold text-gray-900 mb-4">Code Metrics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
            <FileText className="w-4 h-4" />
            <span>Lines</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {metrics.originalLines}
          </div>
          <div className="text-xs text-gray-500">
            → {metrics.improvedLines} ({lineDiff > 0 ? '+' : ''}
            {lineDiff})
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
            <Code className="w-4 h-4" />
            <span>Characters</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {metrics.originalChars.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">
            → {metrics.improvedChars.toLocaleString()} ({charDiff > 0 ? '+' : ''}
            {charDiff})
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
            <AlertCircle className="w-4 h-4" />
            <span>Suggestions</span>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {metrics.suggestionsCount}
          </div>
          <div className="flex gap-1 mt-1">
            {metrics.highSeverity > 0 && (
              <Badge variant="error" size="sm">
                {metrics.highSeverity} High
              </Badge>
            )}
            {metrics.mediumSeverity > 0 && (
              <Badge variant="warning" size="sm">
                {metrics.mediumSeverity} Med
              </Badge>
            )}
            {metrics.lowSeverity > 0 && (
              <Badge variant="info" size="sm">
                {metrics.lowSeverity} Low
              </Badge>
            )}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
            <CheckCircle className="w-4 h-4" />
            <span>Improvement</span>
          </div>
          <div className="text-2xl font-bold text-green-600">
            {metrics.suggestionsCount > 0 ? 'Yes' : 'No'}
          </div>
          <div className="text-xs text-gray-500">
            {metrics.suggestionsCount} changes
          </div>
        </div>
      </div>
    </Card>
  )
}

export default CodeMetrics




