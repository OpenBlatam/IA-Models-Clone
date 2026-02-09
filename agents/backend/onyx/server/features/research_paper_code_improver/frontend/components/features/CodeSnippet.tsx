'use client'

import React, { useState } from 'react'
import { Copy, Check } from 'lucide-react'
import { Button } from '../ui'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import toast from 'react-hot-toast'

interface CodeSnippetProps {
  code: string
  language?: string
  showLineNumbers?: boolean
  startingLineNumber?: number
  className?: string
  title?: string
}

const CodeSnippet: React.FC<CodeSnippetProps> = ({
  code,
  language = 'python',
  showLineNumbers = false,
  startingLineNumber = 1,
  className,
  title,
}) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    toast.success('Code copied to clipboard!')
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={className}>
      {title && (
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-medium text-gray-700">{title}</h4>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-7"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 mr-1" />
                Copied
              </>
            ) : (
              <>
                <Copy className="w-3 h-3 mr-1" />
                Copy
              </>
            )}
          </Button>
        </div>
      )}
      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={vscDarkPlus}
          customStyle={{
            borderRadius: '8px',
            fontSize: '14px',
            margin: 0,
            padding: '1rem',
          }}
          showLineNumbers={showLineNumbers}
          startingLineNumber={startingLineNumber}
        >
          {code}
        </SyntaxHighlighter>
        {!title && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="absolute top-2 right-2 h-7"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-600" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
        )}
      </div>
    </div>
  )
}

export default CodeSnippet




