'use client'

import { useRef, useCallback, useState, useEffect } from 'react'
import { Trash2, Download } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui'
import { fileUtils } from '@/lib/utils'

interface SignaturePadProps {
  onSave?: (dataUrl: string) => void
  width?: number
  height?: number
  className?: string
  penColor?: string
  backgroundColor?: string
}

const SignaturePad = ({
  onSave,
  width = 400,
  height = 200,
  className,
  penColor = '#000000',
  backgroundColor = '#FFFFFF',
}: SignaturePadProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)

  const startDrawing = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top

    ctx.beginPath()
    ctx.moveTo(x, y)
    setIsDrawing(true)
  }, [])

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) {
      return
    }

    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top

    ctx.lineTo(x, y)
    ctx.stroke()
  }, [isDrawing])

  const stopDrawing = useCallback(() => {
    setIsDrawing(false)
  }, [])

  const clear = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }, [backgroundColor])

  const save = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const dataUrl = canvas.toDataURL('image/png')
    onSave?.(dataUrl)
  }, [onSave])

  const download = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const dataUrl = canvas.toDataURL('image/png')
    fileUtils.download(dataUrl, `signature-${Date.now()}.png`, 'image/png')
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = penColor
    ctx.lineWidth = 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }, [backgroundColor, penColor])

  return (
    <div className={cn('space-y-4', className)}>
      <div className="border border-gray-300 rounded-lg overflow-hidden">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="cursor-crosshair w-full"
        />
      </div>
      <div className="flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={clear} leftIcon={<Trash2 className="w-4 h-4" />}>
          Clear
        </Button>
        <Button variant="secondary" size="sm" onClick={download} leftIcon={<Download className="w-4 h-4" />}>
          Download
        </Button>
        {onSave && (
          <Button variant="primary" size="sm" onClick={save}>
            Save
          </Button>
        )}
      </div>
    </div>
  )
}

export default SignaturePad

