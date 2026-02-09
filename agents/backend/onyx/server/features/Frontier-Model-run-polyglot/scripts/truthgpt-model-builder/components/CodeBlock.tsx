'use client'

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import { Copy, Check } from 'lucide-react'
import { useCopyToClipboard } from '@/lib/hooks/useCopyToClipboard'
import { cn } from '@/lib/cn'

interface CodeBlockProps {
  code: string
  language?: string
  className?: string
}

export default function CodeBlock({ code, language = 'python', className }: CodeBlockProps) {
  const { copy, copied } = useCopyToClipboard()

  return (
    <div className={cn('relative group', className)}>
      <button
        onClick={() => copy(code)}
        className="absolute top-2 right-2 p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors opacity-0 group-hover:opacity-100 z-10"
        title="Copiar código"
      >
        {copied ? (
          <Check className="w-4 h-4 text-green-400" />
        ) : (
          <Copy className="w-4 h-4 text-slate-300" />
        )}
      </button>
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        customStyle={{
          borderRadius: '0.5rem',
          padding: '1rem',
          fontSize: '0.875rem',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}

