'use client'

import { TABS, type TabType } from '@/lib/constants'
import { SimpleTabs } from '@/components/ui'

interface NavigationProps {
  activeTab: TabType
  queueSize: number
  onTabChange: (tab: TabType) => void
}

const Navigation = ({ activeTab, queueSize, onTabChange }: NavigationProps) => {
  const tabs = [
    { id: TABS.GENERATE, label: 'Generate', shortcut: '1' },
    { id: TABS.QUEUE, label: `Queue (${queueSize})`, shortcut: '2' },
    { id: TABS.PROJECTS, label: 'Projects', shortcut: '3' },
    { id: TABS.STATS, label: 'Statistics', shortcut: '4' },
  ] as const

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SimpleTabs tabs={tabs} activeTab={activeTab} onTabChange={onTabChange} />
      </div>
    </nav>
  )
}

export default Navigation

