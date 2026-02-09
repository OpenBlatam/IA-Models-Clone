'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import {
  LayoutDashboard,
  FileText,
  Brain,
  Code,
  History,
  Settings,
  FileCode,
  Github,
} from 'lucide-react'
import { CommandPalette } from '../ui'

const GlobalCommandPalette: React.FC = () => {
  const router = useRouter()

  const commands = [
    {
      id: 'dashboard',
      label: 'Go to Dashboard',
      description: 'Navigate to the main dashboard',
      icon: <LayoutDashboard className="w-4 h-4" />,
      action: () => router.push('/'),
      keywords: ['home', 'main'],
      category: 'Navigation',
    },
    {
      id: 'papers',
      label: 'Go to Papers',
      description: 'View and manage research papers',
      icon: <FileText className="w-4 h-4" />,
      action: () => router.push('/papers'),
      keywords: ['papers', 'documents'],
      category: 'Navigation',
    },
    {
      id: 'training',
      label: 'Go to Training',
      description: 'Train models with research papers',
      icon: <Brain className="w-4 h-4" />,
      action: () => router.push('/training'),
      keywords: ['train', 'model'],
      category: 'Navigation',
    },
    {
      id: 'code-improve',
      label: 'Improve Code',
      description: 'Improve code using AI models',
      icon: <Code className="w-4 h-4" />,
      action: () => router.push('/code-improve'),
      keywords: ['improve', 'code'],
      category: 'Navigation',
    },
    {
      id: 'batch',
      label: 'Batch Process',
      description: 'Process multiple files at once',
      icon: <FileCode className="w-4 h-4" />,
      action: () => router.push('/batch'),
      keywords: ['batch', 'multiple'],
      category: 'Navigation',
    },
    {
      id: 'analyze',
      label: 'Analyze Repository',
      description: 'Analyze GitHub repositories',
      icon: <Github className="w-4 h-4" />,
      action: () => router.push('/analyze'),
      keywords: ['analyze', 'repo', 'github'],
      category: 'Navigation',
    },
    {
      id: 'history',
      label: 'View History',
      description: 'View code improvement history',
      icon: <History className="w-4 h-4" />,
      action: () => router.push('/history'),
      keywords: ['history', 'past'],
      category: 'Navigation',
    },
    {
      id: 'settings',
      label: 'Go to Settings',
      description: 'Configure application settings',
      icon: <Settings className="w-4 h-4" />,
      action: () => router.push('/settings'),
      keywords: ['settings', 'config'],
      category: 'Navigation',
    },
  ]

  return <CommandPalette commands={commands} />
}

export default GlobalCommandPalette



