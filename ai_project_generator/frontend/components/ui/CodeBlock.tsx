'use client'

import { useState, useCallback } from 'react'
import { Copy, Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useCopyToClipboard } from '@/hooks/ui'

interface CodeBlockProps {
  code: string
  language?: string
  className?: string
  showLineNumbers?: boolean
  highlightLines?: number[]
}

const CodeBlock = ({
  code,
  language,
  className,
  showLineNumbers = false,
  highlightLines = [],
}: CodeBlockProps) => {
  const { copy, copied } = useCopyToClipboard()
  const [hovered, setHovered] = useState(false)

  const handleCopy = useCallback(() => {
    copy(code)
  }, [code, copy])

  const lines = code.split('\n')

  return (
    <div
      className={cn('relative rounded-lg bg-gray-900 text-gray-100', className)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {language && (
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
          <span className="text-xs font-medium text-gray-400 uppercase">{language}</span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-1 text-xs hover:bg-gray-800 rounded transition-colors"
            aria-label="Copy code"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3" />
                <span>Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      )}
      <div className="overflow-x-auto p-4">
        <pre className="text-sm">
          <code>
            {lines.map((line, index) => (
              <div
                key={index}
                className={cn(
                  'flex',
                  showLineNumbers && 'gap-4',
                  highlightLines.includes(index + 1) && 'bg-yellow-900/30'
                )}
              >
                {showLineNumbers && (
                  <span className="text-gray-500 select-none text-right min-w-[3ch]">
                    {index + 1}
                  </span>
                )}
                <span>{line || '\u00A0'}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  )
}

export default CodeBlock

