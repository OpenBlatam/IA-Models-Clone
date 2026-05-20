'use client'

import { useState, ReactNode, useCallback } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button, Stepper } from '@/components/ui'

interface WizardStep {
  id: string
  title: string
  description?: string
  content: ReactNode
  optional?: boolean
}

interface WizardProps {
  steps: WizardStep[]
  onComplete?: () => void
  onStepChange?: (stepIndex: number) => void
  className?: string
  showStepper?: boolean
  allowSkip?: boolean
}

const Wizard = ({
  steps,
  onComplete,
  onStepChange,
  className,
  showStepper = true,
  allowSkip = false,
}: WizardProps) => {
  const [currentStep, setCurrentStep] = useState(0)

  const handleNext = useCallback(() => {
    if (currentStep < steps.length - 1) {
      const newStep = currentStep + 1
      setCurrentStep(newStep)
      onStepChange?.(newStep)
    } else {
      onComplete?.()
    }
  }, [currentStep, steps.length, onComplete, onStepChange])

  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      const newStep = currentStep - 1
      setCurrentStep(newStep)
      onStepChange?.(newStep)
    }
  }, [currentStep, onStepChange])

  const handleStepClick = useCallback(
    (stepIndex: number) => {
      if (allowSkip || stepIndex <= currentStep) {
        setCurrentStep(stepIndex)
        onStepChange?.(stepIndex)
      }
    },
    [currentStep, allowSkip, onStepChange]
  )

  const currentStepData = steps[currentStep]

  return (
    <div className={cn('w-full', className)}>
      {showStepper && (
        <div className="mb-8">
          <Stepper
            steps={steps.map((step) => ({
              id: step.id,
              label: step.title,
              description: step.description,
            }))}
            currentStep={currentStep}
            orientation="horizontal"
          />
        </div>
      )}

      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">{currentStepData.title}</h2>
        {currentStepData.description && (
          <p className="text-gray-600">{currentStepData.description}</p>
        )}
      </div>

      <div className="min-h-[300px] mb-6">{currentStepData.content}</div>

      <div className="flex items-center justify-between pt-6 border-t border-gray-200">
        <Button
          variant="secondary"
          onClick={handlePrevious}
          disabled={currentStep === 0}
          leftIcon={<ChevronLeft className="w-4 h-4" />}
        >
          Previous
        </Button>

        <div className="text-sm text-gray-500">
          Step {currentStep + 1} of {steps.length}
        </div>

        <Button
          variant="primary"
          onClick={handleNext}
          rightIcon={currentStep < steps.length - 1 ? <ChevronRight className="w-4 h-4" /> : undefined}
        >
          {currentStep < steps.length - 1 ? 'Next' : 'Complete'}
        </Button>
      </div>
    </div>
  )
}

export default Wizard

