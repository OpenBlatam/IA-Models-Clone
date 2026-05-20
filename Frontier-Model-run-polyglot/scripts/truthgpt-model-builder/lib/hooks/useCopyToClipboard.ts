import { useState } from 'react'
import { toast } from 'react-hot-toast'

export function useCopyToClipboard() {
  const [copied, setCopied] = useState(false)

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast.success('Copiado al portapapeles')
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      toast.error('Error al copiar')
      console.error('Failed to copy:', err)
    }
  }

  return { copy, copied }
}


