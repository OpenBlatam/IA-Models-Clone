'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import CodeBlock from './CodeBlock'

interface MarkdownProps {
  content: string
  className?: string
}

const Markdown = ({ content, className }: MarkdownProps) => {
  const parseMarkdown = (text: string): ReactNode[] => {
    const lines = text.split('\n')
    const elements: ReactNode[] = []
    let currentParagraph: string[] = []
    let inCodeBlock = false
    let codeBlockContent: string[] = []
    let codeBlockLanguage = ''

    const flushParagraph = () => {
      if (currentParagraph.length > 0) {
        const paragraphText = currentParagraph.join(' ')
        if (paragraphText.trim()) {
          elements.push(
            <p key={elements.length} className="mb-4 text-gray-700">
              {parseInlineMarkdown(paragraphText)}
            </p>
          )
        }
        currentParagraph = []
      }
    }

    const parseInlineMarkdown = (text: string): ReactNode => {
      const parts: ReactNode[] = []
      let currentIndex = 0

      const patterns = [
        { regex: /\*\*(.*?)\*\*/g, render: (match: string) => <strong key={currentIndex++}>{match}</strong> },
        { regex: /\*(.*?)\*/g, render: (match: string) => <em key={currentIndex++}>{match}</em> },
        { regex: /`(.*?)`/g, render: (match: string) => (
          <code key={currentIndex++} className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">
            {match}
          </code>
        )},
        { regex: /\[([^\]]+)\]\(([^)]+)\)/g, render: (match: string, link: string, url: string) => (
          <a key={currentIndex++} href={url} className="text-primary-600 hover:underline">
            {link}
          </a>
        )},
      ]

      let lastIndex = 0
      const matches: Array<{ start: number; end: number; render: () => ReactNode }> = []

      patterns.forEach((pattern) => {
        let match
        while ((match = pattern.regex.exec(text)) !== null) {
          matches.push({
            start: match.index,
            end: match.index + match[0].length,
            render: () => pattern.render(match[1] || match[2] || match[0]),
          })
        }
      })

      matches.sort((a, b) => a.start - b.start)

      matches.forEach((match) => {
        if (match.start > lastIndex) {
          parts.push(text.substring(lastIndex, match.start))
        }
        parts.push(match.render())
        lastIndex = match.end
      })

      if (lastIndex < text.length) {
        parts.push(text.substring(lastIndex))
      }

      return parts.length > 0 ? <>{parts}</> : text
    }

    lines.forEach((line, index) => {
      if (line.startsWith('```')) {
        if (inCodeBlock) {
          flushParagraph()
          elements.push(
            <CodeBlock
              key={elements.length}
              code={codeBlockContent.join('\n')}
              language={codeBlockLanguage}
              className="my-4"
            />
          )
          codeBlockContent = []
          codeBlockLanguage = ''
          inCodeBlock = false
        } else {
          flushParagraph()
          codeBlockLanguage = line.substring(3).trim()
          inCodeBlock = true
        }
        return
      }

      if (inCodeBlock) {
        codeBlockContent.push(line)
        return
      }

      if (line.startsWith('# ')) {
        flushParagraph()
        elements.push(
          <h1 key={elements.length} className="text-3xl font-bold mb-4 text-gray-900">
            {parseInlineMarkdown(line.substring(2))}
          </h1>
        )
      } else if (line.startsWith('## ')) {
        flushParagraph()
        elements.push(
          <h2 key={elements.length} className="text-2xl font-bold mb-3 text-gray-900">
            {parseInlineMarkdown(line.substring(3))}
          </h2>
        )
      } else if (line.startsWith('### ')) {
        flushParagraph()
        elements.push(
          <h3 key={elements.length} className="text-xl font-bold mb-2 text-gray-900">
            {parseInlineMarkdown(line.substring(4))}
          </h3>
        )
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        flushParagraph()
        elements.push(
          <li key={elements.length} className="ml-4 mb-1 text-gray-700">
            {parseInlineMarkdown(line.substring(2))}
          </li>
        )
      } else if (line.trim() === '') {
        flushParagraph()
      } else {
        currentParagraph.push(line)
      }
    })

    flushParagraph()

    return elements
  }

  return <div className={cn('prose max-w-none', className)}>{parseMarkdown(content)}</div>
}

export default Markdown

