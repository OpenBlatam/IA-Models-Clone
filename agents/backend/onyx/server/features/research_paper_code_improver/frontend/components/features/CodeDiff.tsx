'use client'

import React from 'react'
import { FileCode, Plus, Minus } from 'lucide-react'
import { Card } from '../ui'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface CodeDiffProps {
  original: string
  improved: string
  language?: string
}

const CodeDiff: React.FC<CodeDiffProps> = ({
  original,
  improved,
  language = 'python',
}) => {
  const originalLines = original.split('\n')
  const improvedLines = improved.split('\n')
  const maxLines = Math.max(originalLines.length, improvedLines.length)

  const getLineDiff = (lineIndex: number) => {
    const origLine = originalLines[lineIndex] || ''
    const imprLine = improvedLines[lineIndex] || ''

    if (origLine === imprLine) {
      return 'same'
    }
    if (!origLine && imprLine) {
      return 'added'
    }
    if (origLine && !imprLine) {
      return 'removed'
    }
    return 'modified'
  }

  return (
    <Card>
      <div className="flex items-center gap-2 mb-4">
        <FileCode className="w-5 h-5 text-primary-600" />
        <h3 className="font-semibold text-gray-900">Code Diff View</h3>
      </div>

      <div className="overflow-x-auto">
        <div className="grid grid-cols-2 gap-4 font-mono text-sm">
          <div>
            <div className="bg-red-50 border border-red-200 rounded-t p-2 font-semibold text-red-800">
              Original Code
            </div>
            <div className="border border-red-200 border-t-0 rounded-b overflow-hidden">
              {originalLines.map((line, index) => {
                const diffType = getLineDiff(index)
                return (
                  <div
                    key={index}
                    className={`
                      px-3 py-1 border-b border-gray-200 last:border-b-0
                      ${diffType === 'removed' ? 'bg-red-50' : ''}
                      ${diffType === 'modified' ? 'bg-yellow-50' : ''}
                    `}
                  >
                    <span className="text-gray-500 mr-2 select-none">
                      {index + 1}
                    </span>
                    <span
                      className={
                        diffType === 'removed' || diffType === 'modified'
                          ? 'text-red-800 line-through'
                          : 'text-gray-800'
                      }
                    >
                      {line || '\u00A0'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          <div>
            <div className="bg-green-50 border border-green-200 rounded-t p-2 font-semibold text-green-800">
              Improved Code
            </div>
            <div className="border border-green-200 border-t-0 rounded-b overflow-hidden">
              {improvedLines.map((line, index) => {
                const diffType = getLineDiff(index)
                return (
                  <div
                    key={index}
                    className={`
                      px-3 py-1 border-b border-gray-200 last:border-b-0
                      ${diffType === 'added' ? 'bg-green-50' : ''}
                      ${diffType === 'modified' ? 'bg-yellow-50' : ''}
                    `}
                  >
                    <span className="text-gray-500 mr-2 select-none">
                      {index + 1}
                    </span>
                    <span
                      className={
                        diffType === 'added' || diffType === 'modified'
                          ? 'text-green-800 font-semibold'
                          : 'text-gray-800'
                      }
                    >
                      {line || '\u00A0'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}

export default CodeDiff




