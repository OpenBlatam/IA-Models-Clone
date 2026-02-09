'use client'

import React from 'react'
import { Check } from 'lucide-react'

interface Step {
  id: string
  label: string
  description?: string
}

interface StepperProps {
  steps: Step[]
  currentStep: number
  onStepClick?: (stepIndex: number) => void
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

const Stepper: React.FC<StepperProps> = ({
  steps,
  currentStep,
  onStepClick,
  orientation = 'horizontal',
  className = '',
}) => {
  const getStepStatus = (index: number) => {
    if (index < currentStep) return 'completed'
    if (index === currentStep) return 'current'
    return 'upcoming'
  }

  if (orientation === 'vertical') {
    return (
      <div className={`space-y-4 ${className}`}>
        {steps.map((step, index) => {
          const status = getStepStatus(index)
          const isClickable = onStepClick && status !== 'upcoming'

          return (
            <div key={step.id} className="flex gap-4">
              <div className="flex flex-col items-center">
                <button
                  onClick={() => isClickable && onStepClick?.(index)}
                  disabled={!isClickable}
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-colors ${
                    status === 'completed'
                      ? 'bg-primary-600 text-white'
                      : status === 'current'
                      ? 'bg-primary-100 text-primary-700 border-2 border-primary-600'
                      : 'bg-gray-200 text-gray-500'
                  } ${isClickable ? 'cursor-pointer hover:opacity-80' : 'cursor-default'}`}
                  aria-label={`Step ${index + 1}: ${step.label}`}
                >
                  {status === 'completed' ? (
                    <Check className="w-5 h-5" />
                  ) : (
                    index + 1
                  )}
                </button>
                {index < steps.length - 1 && (
                  <div
                    className={`w-0.5 h-full min-h-8 ${
                      status === 'completed' ? 'bg-primary-600' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
              <div className="flex-1 pb-4">
                <h3
                  className={`font-semibold ${
                    status === 'current'
                      ? 'text-primary-700'
                      : status === 'completed'
                      ? 'text-gray-900'
                      : 'text-gray-500'
                  }`}
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
    <nav
      aria-label="Progress"
      className={`flex items-center justify-between ${className}`}
    >
      {steps.map((step, index) => {
        const status = getStepStatus(index)
        const isClickable = onStepClick && status !== 'upcoming'

        return (
          <React.Fragment key={step.id}>
            <div className="flex items-center">
              <button
                onClick={() => isClickable && onStepClick?.(index)}
                disabled={!isClickable}
                className={`flex flex-col items-center ${
                  isClickable ? 'cursor-pointer' : 'cursor-default'
                }`}
                aria-label={`Step ${index + 1}: ${step.label}`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-colors ${
                    status === 'completed'
                      ? 'bg-primary-600 text-white'
                      : status === 'current'
                      ? 'bg-primary-100 text-primary-700 border-2 border-primary-600'
                      : 'bg-gray-200 text-gray-500'
                  }`}
                >
                  {status === 'completed' ? (
                    <Check className="w-5 h-5" />
                  ) : (
                    index + 1
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p
                    className={`text-sm font-medium ${
                      status === 'current'
                        ? 'text-primary-700'
                        : status === 'completed'
                        ? 'text-gray-900'
                        : 'text-gray-500'
                    }`}
                  >
                    {step.label}
                  </p>
                  {step.description && (
                    <p className="text-xs text-gray-500 mt-1">
                      {step.description}
                    </p>
                  )}
                </div>
              </button>
            </div>
            {index < steps.length - 1 && (
              <div
                className={`flex-1 h-0.5 mx-4 ${
                  index < currentStep ? 'bg-primary-600' : 'bg-gray-200'
                }`}
                aria-hidden="true"
              />
            )}
          </React.Fragment>
        )
      })}
    </nav>
  )
}

export default Stepper



