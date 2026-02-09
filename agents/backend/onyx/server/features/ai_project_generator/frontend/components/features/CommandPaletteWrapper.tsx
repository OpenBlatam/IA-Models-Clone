'use client'

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { CommandPalette } from '@/components/ui'
import { TABS } from '@/lib/constants'
import { Sparkles, List, BarChart3, Clock } from 'lucide-react'

interface CommandPaletteWrapperProps {
  onTabChange: (tab: string) => void
}

const CommandPaletteWrapper = ({ onTabChange }: CommandPaletteWrapperProps) => {
  const router = useRouter()

  const commands = [
    {
      id: 'generate',
      label: 'Go to Generate',
      icon: <Sparkles className="w-4 h-4" />,
      action: () => onTabChange(TABS.GENERATE),
      keywords: ['generate', 'create', 'new'],
    },
    {
      id: 'queue',
      label: 'Go to Queue',
      icon: <Clock className="w-4 h-4" />,
      action: () => onTabChange(TABS.QUEUE),
      keywords: ['queue', 'pending', 'waiting'],
    },
    {
      id: 'projects',
      label: 'Go to Projects',
      icon: <List className="w-4 h-4" />,
      action: () => onTabChange(TABS.PROJECTS),
      keywords: ['projects', 'list', 'all'],
    },
    {
      id: 'stats',
      label: 'Go to Statistics',
      icon: <BarChart3 className="w-4 h-4" />,
      action: () => onTabChange(TABS.STATS),
      keywords: ['stats', 'statistics', 'analytics'],
    },
  ]

  return <CommandPalette items={commands} />
}

export default CommandPaletteWrapper

