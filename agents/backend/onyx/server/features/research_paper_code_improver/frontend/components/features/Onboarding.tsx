'use client'

import React, { useState, useEffect } from 'react'
import { X, ChevronRight, ChevronLeft, Check } from 'lucide-react'
import { Button, Card, ProgressBar } from '../ui'
import { storage, storageKeys } from '@/lib/utils'

interface OnboardingStep {
  title: string
  description: string
  content: React.ReactNode
}

interface OnboardingProps {
  steps: OnboardingStep[]
  storageKey?: string
  onComplete?: () => void
}

const Onboarding: React.FC<OnboardingProps> = ({
  steps,
  storageKey = 'onboarding-completed',
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const completed = storage.get<boolean>(storageKey, false)
    if (!completed) {
      setIsVisible(true)
    }
  }, [storageKey])

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      handleComplete()
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleComplete = () => {
    storage.set(storageKey, true)
    setIsVisible(false)
    onComplete?.()
  }

  const handleSkip = () => {
    handleComplete()
  }

  if (!isVisible) return null

  const progress = ((currentStep + 1) / steps.length) * 100

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 animate-fade-in">
      <Card className="max-w-2xl w-full mx-4 animate-slide-up">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">
              {steps[currentStep].title}
            </h2>
            <button
              onClick={handleSkip}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="Skip onboarding"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          <ProgressBar value={progress} showPercentage />

          <div className="min-h-[300px]">
            {steps[currentStep].content}
          </div>

          <p className="text-sm text-gray-600">
            {steps[currentStep].description}
          </p>

          <div className="flex items-center justify-between pt-4 border-t border-gray-200">
            <Button
              variant="ghost"
              onClick={handlePrevious}
              disabled={currentStep === 0}
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              Previous
            </Button>

            <div className="flex gap-2">
              {steps.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full ${
                    index === currentStep
                      ? 'bg-primary-600'
                      : index < currentStep
                      ? 'bg-primary-300'
                      : 'bg-gray-300'
                  }`}
                />
              ))}
            </div>

            <Button onClick={handleNext}>
              {currentStep === steps.length - 1 ? (
                <>
                  Complete
                  <Check className="w-4 h-4 ml-1" />
                </>
              ) : (
                <>
                  Next
                  <ChevronRight className="w-4 h-4 ml-1" />
                </>
              )}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default Onboarding




