'use client'

import type { Stats } from '@/types'
import { BarChart3, CheckCircle, XCircle, Clock, TrendingUp } from 'lucide-react'
import { LoadingSpinner, ProgressBar, Card, CountUp } from '@/components/ui'

interface StatisticsProps {
  stats: Stats | null
}

const Statistics = ({ stats }: StatisticsProps) => {
  if (!stats) {
    return (
      <Card>
        <LoadingSpinner size="lg" text="Loading statistics..." />
      </Card>
    )
  }

  const completionRate =
    stats.total_projects > 0
      ? ((stats.completed_projects / stats.total_projects) * 100).toFixed(1)
      : '0'

  const failureRate =
    stats.total_projects > 0
      ? ((stats.failed_projects / stats.total_projects) * 100).toFixed(1)
      : '0'

  return (
    <div className="space-y-6">
      <Card>
        <div className="flex items-center gap-2 mb-6">
          <BarChart3 className="w-6 h-6 text-primary-600" aria-hidden="true" />
          <h2 className="text-2xl font-bold text-gray-900">Statistics Overview</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-5 h-5 text-blue-600" aria-hidden="true" />
              <h3 className="text-sm font-medium text-blue-900">Total Projects</h3>
            </div>
                <p className="text-3xl font-bold text-blue-600">
                  <CountUp end={stats.total_projects} duration={1500} />
                </p>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-600" aria-hidden="true" />
              <h3 className="text-sm font-medium text-green-900">Completed</h3>
            </div>
                <p className="text-3xl font-bold text-green-600">
                  <CountUp end={stats.completed_projects} duration={1500} />
                </p>
            <p className="text-sm text-green-700 mt-1 mb-2">{completionRate}% success rate</p>
            <ProgressBar
              value={parseFloat(completionRate)}
              color="success"
              size="sm"
              showValue={false}
            />
          </div>

          <div className="bg-red-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <XCircle className="w-5 h-5 text-red-600" aria-hidden="true" />
              <h3 className="text-sm font-medium text-red-900">Failed</h3>
            </div>
                <p className="text-3xl font-bold text-red-600">
                  <CountUp end={stats.failed_projects} duration={1500} />
                </p>
            <p className="text-sm text-red-700 mt-1 mb-2">{failureRate}% failure rate</p>
            <ProgressBar
              value={parseFloat(failureRate)}
              color="danger"
              size="sm"
              showValue={false}
            />
          </div>

          <div className="bg-yellow-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-5 h-5 text-yellow-600" aria-hidden="true" />
              <h3 className="text-sm font-medium text-yellow-900">In Queue</h3>
            </div>
                <p className="text-3xl font-bold text-yellow-600">
                  <CountUp end={stats.queue_size} duration={1500} />
                </p>
          </div>
        </div>

        {stats.average_generation_time && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-5 h-5 text-gray-600" aria-hidden="true" />
              <h3 className="text-sm font-medium text-gray-900">Average Generation Time</h3>
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {stats.average_generation_time.toFixed(2)}s
            </p>
          </div>
        )}
      </Card>

      {stats.projects_by_type && Object.keys(stats.projects_by_type).length > 0 && (
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Projects by AI Type</h3>
          <div className="space-y-3">
            {Object.entries(stats.projects_by_type)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .map(([type, count]) => {
                const total = Object.values(stats.projects_by_type!).reduce(
                  (sum, val) => sum + (val as number),
                  0
                )
                const percentage = total > 0 ? ((count as number) / total) * 100 : 0
                return (
                  <div key={type}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-gray-700 capitalize font-medium">{type}</span>
                      <span className="font-semibold text-gray-900">
                        {count as number} ({percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <ProgressBar
                      value={count as number}
                      max={total}
                      color="primary"
                      size="sm"
                      showValue={false}
                    />
                  </div>
                )
              })}
          </div>
        </Card>
      )}

      {stats.projects_by_framework && Object.keys(stats.projects_by_framework).length > 0 && (
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Projects by Framework</h3>
          <div className="space-y-3">
            {Object.entries(stats.projects_by_framework)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .map(([framework, count]) => {
                const total = Object.values(stats.projects_by_framework!).reduce(
                  (sum, val) => sum + (val as number),
                  0
                )
                const percentage = total > 0 ? ((count as number) / total) * 100 : 0
                return (
                  <div key={framework}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-gray-700 capitalize font-medium">{framework}</span>
                      <span className="font-semibold text-gray-900">
                        {count as number} ({percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <ProgressBar
                      value={count as number}
                      max={total}
                      color="primary"
                      size="sm"
                      showValue={false}
                    />
                  </div>
                )
              })}
          </div>
        </Card>
      )}
    </div>
  )
}

export default Statistics

