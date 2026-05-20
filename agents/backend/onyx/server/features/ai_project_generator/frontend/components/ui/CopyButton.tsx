'use client'

import { useState, useCallback } from 'react'
import { Copy, Check } from 'lucide-react'
import Button from './Button'

interface CopyButtonProps {
  text: string
  label?: string
  size?: 'sm' | 'md' | 'lg'
  variant?: 'primary' | 'secondary' | 'danger'
  className?: string
}

const CopyButton = ({ text, label, size = 'sm', variant = 'secondary', className }: CopyButtonProps) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text:', err)
    }
  }, [text])

  return (
    <Button
      variant={variant}
      size={size}
      leftIcon={copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
      onClick={handleCopy}
      className={className}
      aria-label={label || 'Copy to clipboard'}
    >
      {copied ? 'Copied!' : label || 'Copy'}
    </Button>
  )
}

export default CopyButton

