'use client'

import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

interface QRCodeProps {
  value: string
  size?: number
  className?: string
  errorCorrectionLevel?: 'L' | 'M' | 'Q' | 'H'
}

const QRCode = ({
  value,
  size = 200,
  className,
  errorCorrectionLevel = 'M',
}: QRCodeProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    canvas.width = size
    canvas.height = size

    const qrSize = size
    const moduleSize = qrSize / 25

    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, qrSize, qrSize)

    ctx.fillStyle = '#000000'

    const generateQRPattern = (text: string) => {
      const modules: boolean[][] = []
      for (let i = 0; i < 25; i++) {
        modules[i] = []
        for (let j = 0; j < 25; j++) {
          const hash = (i * 25 + j + text.length) % 3
          modules[i][j] = hash === 0
        }
      }

      const finderPattern = (x: number, y: number) => {
        for (let i = 0; i < 7; i++) {
          for (let j = 0; j < 7; j++) {
            const pattern = [
              [1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 1, 1, 0, 1],
              [1, 0, 1, 1, 1, 0, 1],
              [1, 0, 1, 1, 1, 0, 1],
              [1, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1],
            ]
            if (pattern[i] && pattern[i][j]) {
              modules[x + i] = modules[x + i] || []
              modules[x + i][y + j] = true
            }
          }
        }
      }

      finderPattern(0, 0)
      finderPattern(0, 18)
      finderPattern(18, 0)

      return modules
    }

    const modules = generateQRPattern(value)

    for (let i = 0; i < 25; i++) {
      for (let j = 0; j < 25; j++) {
        if (modules[i] && modules[i][j]) {
          ctx.fillRect(j * moduleSize, i * moduleSize, moduleSize, moduleSize)
        }
      }
    }
  }, [value, size])

  return (
    <div className={cn('inline-block', className)}>
      <canvas ref={canvasRef} className="border border-gray-300 rounded" />
    </div>
  )
}

export default QRCode

