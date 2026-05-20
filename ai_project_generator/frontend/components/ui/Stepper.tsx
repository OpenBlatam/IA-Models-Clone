'use client'

import { ReactNode } from 'react'
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Step {
  id: string
  label: string
  description?: string
  icon?: ReactNode
}

interface StepperProps {
  steps: Step[]
  currentStep: number
  className?: string
  orientation?: 'horizontal' | 'vertical'
}

const Stepper = ({ steps, currentStep, className, orientation = 'horizontal' }: StepperProps) => {
  if (orientation === 'vertical') {
    return (
      <div className={cn('space-y-4', className)}>
        {steps.map((step, index) => {
          const isCompleted = index < currentStep
          const isCurrent = index === currentStep
          const isUpcoming = index > currentStep

          return (
            <div key={step.id} className="flex gap-4">
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors',
                    isCompleted && 'bg-primary-600 border-primary-600',
                    isCurrent && 'bg-primary-100 border-primary-600',
                    isUpcoming && 'bg-white border-gray-300'
                  )}
                >
                  {isCompleted ? (
                    <Check className="w-5 h-5 text-white" />
                  ) : step.icon ? (
                    step.icon
                  ) : (
                    <span className={cn('text-sm font-medium', isCurrent && 'text-primary-600')}>
                      {index + 1}
                    </span>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      'w-0.5 flex-1 mt-2',
                      isCompleted ? 'bg-primary-600' : 'bg-gray-300'
                    )}
                  />
                )}
              </div>
              <div className="flex-1 pb-8">
                <h3
                  className={cn(
                    'font-medium',
                    isCurrent && 'text-primary-600',
                    isCompleted && 'text-gray-900',
                    isUpcoming && 'text-gray-500'
                  )}
                >
                  {step.label}
                </h3>
                {step.description && (
                  <p className="text-sm text-gray-600 mt-1">{step.description}</p>
                )}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const isCompleted = index < currentStep
          const isCurrent = index === currentStep
          const isUpcoming = index > currentStep

          return (
            <div key={step.id} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <div
                  className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors',
                    isCompleted && 'bg-primary-600 border-primary-600',
                    isCurrent && 'bg-primary-100 border-primary-600',
                    isUpcoming && 'bg-white border-gray-300'
                  )}
                >
                  {isCompleted ? (
                    <Check className="w-5 h-5 text-white" />
                  ) : step.icon ? (
                    step.icon
                  ) : (
                    <span className={cn('text-sm font-medium', isCurrent && 'text-primary-600')}>
                      {index + 1}
                    </span>
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p
                    className={cn(
                      'text-sm font-medium',
                      isCurrent && 'text-primary-600',
                      isCompleted && 'text-gray-900',
                      isUpcoming && 'text-gray-500'
                    )}
                  >
                    {step.label}
                  </p>
                  {step.description && (
                    <p className="text-xs text-gray-500 mt-1">{step.description}</p>
                  )}
                </div>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={cn(
                    'h-0.5 flex-1 mx-4',
                    isCompleted ? 'bg-primary-600' : 'bg-gray-300'
                  )}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default Stepper

