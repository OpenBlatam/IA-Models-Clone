'use client'

import { useState, useCallback, useEffect, useRef, ReactNode } from 'react'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'
import { createPortal } from 'react-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui'
import { domUtils } from '@/lib/utils'

interface TourStep {
  id: string
  target?: string
  title: string
  content: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
}

interface TourProps {
  steps: TourStep[]
  isOpen: boolean
  onClose: () => void
  onComplete?: () => void
  className?: string
}

const Tour = ({ steps, isOpen, onClose, onComplete, className }: TourProps) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const [mounted, setMounted] = useState(false)
  const overlayRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!isOpen || !steps[currentStep]?.target) {
      return
    }

    const targetElement = document.querySelector(steps[currentStep].target!)
    if (!targetElement) {
      return
    }

    const updatePosition = () => {
      const rect = targetElement.getBoundingClientRect()
      const step = steps[currentStep]
      const position = step.position || 'bottom'

      let top = 0
      let left = 0

      switch (position) {
        case 'top':
          top = rect.top - 20
          left = rect.left + rect.width / 2
          break
        case 'bottom':
          top = rect.bottom + 20
          left = rect.left + rect.width / 2
          break
        case 'left':
          top = rect.top + rect.height / 2
          left = rect.left - 20
          break
        case 'right':
          top = rect.top + rect.height / 2
          left = rect.right + 20
          break
      }

      setPosition({ top, left })
    }

    updatePosition()
    window.addEventListener('scroll', updatePosition, true)
    window.addEventListener('resize', updatePosition)

    return () => {
      window.removeEventListener('scroll', updatePosition, true)
      window.removeEventListener('resize', updatePosition)
    }
  }, [isOpen, currentStep, steps])

  const handleNext = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      onComplete?.()
      onClose()
    }
  }, [currentStep, steps.length, onComplete, onClose])

  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }, [currentStep])

  if (!isOpen || !mounted) {
    return null
  }

  const currentStepData = steps[currentStep]

  return createPortal(
    <>
      <div
        ref={overlayRef}
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />
      {currentStepData.target && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            top: `${position.top}px`,
            left: `${position.left}px`,
            transform: 'translate(-50%, 0)',
          }}
        >
          <div
            className={cn(
              'bg-white rounded-lg shadow-xl p-6 max-w-sm pointer-events-auto',
              className
            )}
          >
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">{currentStepData.title}</h3>
              <button
                onClick={onClose}
                className="p-1 hover:bg-gray-100 rounded"
                aria-label="Close tour"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            <div className="mb-4 text-gray-700">{currentStepData.content}</div>
            <div className="flex items-center justify-between">
              <Button
                variant="secondary"
                size="sm"
                onClick={handlePrevious}
                disabled={currentStep === 0}
                leftIcon={<ChevronLeft className="w-4 h-4" />}
              >
                Previous
              </Button>
              <div className="text-sm text-gray-500">
                {currentStep + 1} / {steps.length}
              </div>
              <Button
                variant="primary"
                size="sm"
                onClick={handleNext}
                rightIcon={currentStep < steps.length - 1 ? <ChevronRight className="w-4 h-4" /> : undefined}
              >
                {currentStep < steps.length - 1 ? 'Next' : 'Finish'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </>,
    document.body
  )
}

export default Tour

