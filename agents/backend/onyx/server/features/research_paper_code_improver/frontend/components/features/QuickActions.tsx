'use client'

import React from 'react'
import { FileText, Brain, Code, History, Upload, Search, FileCode, Github } from 'lucide-react'
import { Card, Button } from '../ui'
import Link from 'next/link'

interface QuickAction {
  icon: React.ElementType
  label: string
  href: string
  description: string
  color: string
}

const QuickActions: React.FC = () => {
  const actions: QuickAction[] = [
    {
      icon: Upload,
      label: 'Upload Paper',
      href: '/papers',
      description: 'Upload a new research paper',
      color: 'bg-blue-100 text-blue-600',
    },
    {
      icon: Search,
      label: 'Search Papers',
      href: '/papers',
      description: 'Find papers in your library',
      color: 'bg-green-100 text-green-600',
    },
    {
      icon: Brain,
      label: 'Train Model',
      href: '/training',
      description: 'Train AI model with papers',
      color: 'bg-purple-100 text-purple-600',
    },
    {
      icon: Code,
      label: 'Improve Code',
      href: '/code-improve',
      description: 'Enhance your code with AI',
      color: 'bg-orange-100 text-orange-600',
    },
    {
      icon: History,
      label: 'View History',
      href: '/history',
      description: 'See past improvements',
      color: 'bg-indigo-100 text-indigo-600',
    },
    {
      icon: FileCode,
      label: 'Batch Process',
      href: '/batch',
      description: 'Process multiple files',
      color: 'bg-pink-100 text-pink-600',
    },
    {
      icon: Github,
      label: 'Analyze Repo',
      href: '/analyze',
      description: 'Analyze entire repository',
      color: 'bg-teal-100 text-teal-600',
    },
  ]

  return (
    <Card>
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        Quick Actions
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {actions.map((action) => {
          const Icon = action.icon
          return (
            <Link
              key={action.href}
              href={action.href}
              className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:shadow-md transition-all group"
            >
              <div className="flex items-start gap-3">
                <div
                  className={`p-2 rounded-lg ${action.color} group-hover:scale-110 transition-transform`}
                >
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-gray-900 group-hover:text-primary-600 transition-colors">
                    {action.label}
                  </h3>
                  <p className="text-xs text-gray-600 mt-1">
                    {action.description}
                  </p>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </Card>
  )
}

export default QuickActions


