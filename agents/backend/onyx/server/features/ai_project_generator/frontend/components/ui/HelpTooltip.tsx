'use client'

import { HelpCircle } from 'lucide-react'
import Tooltip from './Tooltip'

interface HelpTooltipProps {
  content: string
  className?: string
}

const HelpTooltip = ({ content, className }: HelpTooltipProps) => {
  return (
    <Tooltip content={content} position="top">
      <HelpCircle
        className={`w-4 h-4 text-gray-400 cursor-help hover:text-gray-600 transition-colors ${className || ''}`}
        aria-label="Help"
        role="button"
        tabIndex={0}
      />
    </Tooltip>
  )
}

export default HelpTooltip

