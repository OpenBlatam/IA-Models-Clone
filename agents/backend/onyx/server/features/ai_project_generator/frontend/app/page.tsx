'use client'

import { useState, useCallback } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useDashboardData } from '@/hooks/api/useDashboardData'
import { useProjectGenerator } from '@/hooks/api/useProjectGenerator'
import { useGeneratorControl } from '@/hooks/api/useGeneratorControl'
import { TABS, type TabType } from '@/lib/constants'
import Header from '@/components/layout/Header'
import Navigation from '@/components/layout/Navigation'
import { ProjectGeneratorForm, ProjectQueue, Statistics, ProjectList } from '@/components/features'
import { LoadingSpinner, ErrorMessage, BackToTop } from '@/components/ui'
import CommandPaletteWrapper from '@/components/features/CommandPaletteWrapper'
import { useKeyboardShortcut } from '@/hooks/ui'
import toast from 'react-hot-toast'

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>(TABS.GENERATE)
  const [dashboardError, setDashboardError] = useState<string | null>(null)

  const { isConnected } = useWebSocket()
  const { status, queue, stats, loading, error, refresh } = useDashboardData()
  const { generate, loading: generating, error: generateError, clearError: clearGenerateError } =
    useProjectGenerator()
  const {
    start: startGenerator,
    stop: stopGenerator,
    error: controlError,
    clearError: clearControlError,
  } = useGeneratorControl()

  const handleGenerate = useCallback(
    async (request: Parameters<typeof generate>[0]) => {
      const projectId = await generate(request)
      if (projectId) {
        await refresh()
        setActiveTab(TABS.QUEUE)
        toast.success('Project added to queue successfully!')
      }
    },
    [generate, refresh]
  )

  const handleStart = useCallback(async () => {
    const result = await startGenerator()
    if (result) {
      await refresh()
      toast.success('Generator started successfully!')
    }
  }, [startGenerator, refresh])

  const handleStop = useCallback(async () => {
    const result = await stopGenerator()
    if (result) {
      await refresh()
      toast.success('Generator stopped successfully!')
    }
  }, [stopGenerator, refresh])

  const handleTabChange = useCallback((tab: TabType) => {
    setActiveTab(tab)
    setDashboardError(null)
    clearGenerateError()
    clearControlError()
  }, [clearGenerateError, clearControlError])

  const handleDismissError = useCallback(() => {
    setDashboardError(null)
  }, [])

  useKeyboardShortcut({
    key: '1',
    callback: () => setActiveTab(TABS.GENERATE),
    enabled: true,
  })

  useKeyboardShortcut({
    key: '2',
    callback: () => setActiveTab(TABS.QUEUE),
    enabled: true,
  })

  useKeyboardShortcut({
    key: '3',
    callback: () => setActiveTab(TABS.PROJECTS),
    enabled: true,
  })

  useKeyboardShortcut({
    key: '4',
    callback: () => setActiveTab(TABS.STATS),
    enabled: true,
  })

  const displayError = dashboardError || error || generateError || controlError

  if (loading && !status) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading dashboard..." />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        status={{
          isRunning: status?.is_running ?? false,
          isConnected,
          queueSize: status?.queue_size ?? 0,
        }}
        onStart={handleStart}
        onStop={handleStop}
      />

      <Navigation
        activeTab={activeTab}
        queueSize={queue?.queue_size ?? 0}
        onTabChange={handleTabChange}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {displayError && (
          <div className="mb-6">
            <ErrorMessage message={displayError} onDismiss={handleDismissError} />
          </div>
        )}
        {activeTab === TABS.GENERATE && (
          <ProjectGeneratorForm onSubmit={handleGenerate} loading={generating} />
        )}
        {activeTab === TABS.QUEUE && <ProjectQueue queue={queue} onRefresh={refresh} />}
        {activeTab === TABS.PROJECTS && <ProjectList />}
        {activeTab === TABS.STATS && <Statistics stats={stats} />}
      </main>
      <BackToTop />
      <CommandPaletteWrapper onTabChange={handleTabChange} />
    </div>
  )
}

