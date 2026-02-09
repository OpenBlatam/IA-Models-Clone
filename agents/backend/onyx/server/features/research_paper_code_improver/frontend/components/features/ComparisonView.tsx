'use client'

import React, { useState } from 'react'
import { GitCompare, Maximize2, Minimize2 } from 'lucide-react'
import { Card, Button, Tabs, TabsList, TabsTrigger, TabsContent } from '../ui'
import CodeSnippet from './CodeSnippet'
import CodeMetrics from './CodeMetrics'
import CodeDiff from './CodeDiff'

interface ComparisonViewProps {
  original: string
  improved: string
  suggestions: Array<{
    type: string
    description: string
    line?: number
    severity?: string
  }>
  language?: string
}

const ComparisonView: React.FC<ComparisonViewProps> = ({
  original,
  improved,
  suggestions,
  language = 'python',
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [activeTab, setActiveTab] = useState('comparison')

  const containerClass = isFullscreen
    ? 'fixed inset-0 z-50 bg-white p-6 overflow-auto'
    : ''

  const content = (
    <div className={containerClass}>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitCompare className="w-6 h-6 text-primary-600" />
            <h2 className="text-xl font-semibold text-gray-900">
              Code Comparison
            </h2>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            {isFullscreen ? (
              <>
                <Minimize2 className="w-4 h-4 mr-2" />
                Exit Fullscreen
              </>
            ) : (
              <>
                <Maximize2 className="w-4 h-4 mr-2" />
                Fullscreen
              </>
            )}
          </Button>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="comparison">Side by Side</TabsTrigger>
            <TabsTrigger value="diff">Diff View</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          </TabsList>

          <TabsContent value="comparison">
            <div className="grid md:grid-cols-2 gap-6">
              <CodeSnippet
                code={original}
                language={language}
                title="Original Code"
                showLineNumbers
              />
              <CodeSnippet
                code={improved}
                language={language}
                title="Improved Code"
                showLineNumbers
              />
            </div>
          </TabsContent>

          <TabsContent value="diff">
            <CodeDiff
              original={original}
              improved={improved}
              language={language}
            />
          </TabsContent>

          <TabsContent value="metrics">
            <CodeMetrics
              original={original}
              improved={improved}
              suggestions={suggestions}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )

  if (isFullscreen) {
    return content
  }

  return <Card>{content}</Card>
}

export default ComparisonView



