'use client'

import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { FileText, Brain, Code, TrendingUp } from 'lucide-react'
import PageLayout from '@/components/layout/PageLayout'
import { Card, Badge, LoadingSpinner } from '@/components/ui'
import { apiClient } from '@/lib/api'
import PaperUpload from '@/components/features/PaperUpload'
import StatisticsChart from '@/components/features/StatisticsChart'
import QuickActions from '@/components/features/QuickActions'
import Onboarding from '@/components/features/Onboarding'
import StatsCard from '@/components/features/StatsCard'
import AnalyticsDashboard from '@/components/features/AnalyticsDashboard'

const Dashboard: React.FC = () => {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => apiClient.getMetricsStats(24),
    refetchInterval: 60000, // Refetch every minute
  })

  const { data: vectorStats } = useQuery({
    queryKey: ['vector-stats'],
    queryFn: () => apiClient.getVectorStoreStats(),
  })

  const stats = [
    {
      name: 'Papers Indexed',
      value: health?.paper_storage.total_papers || 0,
      icon: FileText,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Vector Store',
      value: vectorStats?.papers_indexed || 0,
      icon: Brain,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Improvements',
      value: metrics?.improvements.total || 0,
      icon: Code,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      name: 'Success Rate',
      value: metrics
        ? `${Math.round((metrics.requests.successful / metrics.requests.total) * 100)}%`
        : '0%',
      icon: TrendingUp,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
    },
  ]

  const handleUploadSuccess = () => {
    // Refetch data after upload
    window.location.reload()
  }

  return (
    <PageLayout>
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
            <p className="mt-2 text-gray-600">
              Monitor your research papers and code improvements
            </p>
          </div>
          <PaperUpload onUploadSuccess={handleUploadSuccess} />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <StatsCard
              key={stat.name}
              title={stat.name}
              value={
                healthLoading || metricsLoading ? (
                  <LoadingSpinner size="sm" />
                ) : (
                  stat.value
                )
              }
              icon={stat.icon}
              color={stat.color}
              bgColor={stat.bgColor}
            />
          ))}
        </div>

        {/* System Status */}
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                System Status
              </h2>
              <p className="mt-1 text-sm text-gray-600">
                Current system health and feature availability
              </p>
            </div>
            <Badge
              variant={health?.status === 'healthy' ? 'success' : 'error'}
              size="lg"
            >
              {health?.status === 'healthy' ? 'Healthy' : 'Issues'}
            </Badge>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <p className="text-sm text-gray-600">RAG Engine</p>
              <Badge
                variant={health?.features.rag ? 'success' : 'error'}
                size="sm"
                className="mt-1"
              >
                {health?.features.rag ? 'Available' : 'Unavailable'}
              </Badge>
            </div>
            <div>
              <p className="text-sm text-gray-600">Cache</p>
              <Badge
                variant={health?.features.cache ? 'success' : 'error'}
                size="sm"
                className="mt-1"
              >
                {health?.features.cache ? 'Available' : 'Unavailable'}
              </Badge>
            </div>
            <div>
              <p className="text-sm text-gray-600">Code Analyzer</p>
              <Badge
                variant={health?.features.analyzer ? 'success' : 'error'}
                size="sm"
                className="mt-1"
              >
                {health?.features.analyzer ? 'Available' : 'Unavailable'}
              </Badge>
            </div>
          </div>
        </Card>

        {/* Analytics Dashboard */}
        {metrics && (
          <AnalyticsDashboard
            metrics={{
              totalImprovements: metrics.improvements?.total || 0,
              averageTime: metrics.improvements?.average_time_ms || 0,
              successRate: metrics.requests
                ? Math.round(
                    (metrics.requests.successful / metrics.requests.total) * 100
                  )
                : 0,
              papersProcessed: health?.paper_storage.total_papers || 0,
            }}
          />
        )}

        {/* Quick Actions */}
        <QuickActions />
      </div>

      {/* Onboarding */}
      <Onboarding
        steps={[
          {
            title: 'Welcome to Research Paper Code Improver',
            description: 'Learn how to get started',
            content: (
              <div className="space-y-4">
                <p className="text-gray-700">
                  This tool helps you improve your code using knowledge from
                  research papers. Upload papers, train models, and enhance your
                  code with AI-powered suggestions.
                </p>
                <div className="bg-primary-50 p-4 rounded-lg">
                  <p className="text-sm text-primary-800">
                    <strong>Tip:</strong> Start by uploading a research paper
                    PDF or processing a paper from a URL.
                  </p>
                </div>
              </div>
            ),
          },
          {
            title: 'Upload Research Papers',
            description: 'Add papers to your library',
            content: (
              <div className="space-y-4">
                <p className="text-gray-700">
                  You can upload papers in two ways:
                </p>
                <ul className="list-disc list-inside space-y-2 text-gray-700">
                  <li>Upload a PDF file directly</li>
                  <li>Process a paper from a URL (arXiv, etc.)</li>
                </ul>
                <p className="text-sm text-gray-600">
                  Papers are automatically indexed and made searchable.
                </p>
              </div>
            ),
          },
          {
            title: 'Train AI Models',
            description: 'Create custom models from your papers',
            content: (
              <div className="space-y-4">
                <p className="text-gray-700">
                  Train AI models using your uploaded papers to get better code
                  improvement suggestions.
                </p>
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    <strong>Note:</strong> Training may take several minutes
                    depending on the number of papers.
                  </p>
                </div>
              </div>
            ),
          },
          {
            title: 'Improve Your Code',
            description: 'Enhance code with AI suggestions',
            content: (
              <div className="space-y-4">
                <p className="text-gray-700">
                  Improve code from GitHub repositories or paste code directly.
                  The AI will suggest improvements based on research papers.
                </p>
                <ul className="list-disc list-inside space-y-2 text-gray-700">
                  <li>View side-by-side comparison</li>
                  <li>See detailed suggestions</li>
                  <li>Export results in multiple formats</li>
                </ul>
              </div>
            ),
          },
        ]}
      />
    </PageLayout>
  )
}

export default Dashboard

