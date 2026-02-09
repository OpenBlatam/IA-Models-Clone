import React from 'react'
import { clsx } from 'clsx'
import { Check } from 'lucide-react'

interface CheckboxProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  error?: string
  helperText?: string
}

const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, error, helperText, ...props }, ref) => {
    return (
      <div className="w-full">
        <label className="flex items-start gap-3 cursor-pointer group">
          <div className="relative flex items-center justify-center flex-shrink-0">
            <input
              ref={ref}
              type="checkbox"
              className="sr-only"
              {...props}
            />
            <div
              className={clsx(
                'w-5 h-5 border-2 rounded transition-all flex items-center justify-center',
                props.checked
                  ? 'bg-primary-600 border-primary-600'
                  : 'border-gray-300 group-hover:border-primary-400',
                error && 'border-red-500',
                props.disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              {props.checked && (
                <Check className="w-4 h-4 text-white" strokeWidth={3} />
              )}
            </div>
          </div>
          {label && (
            <div className="flex-1">
              <span
                className={clsx(
                  'text-sm font-medium',
                  props.checked ? 'text-gray-900' : 'text-gray-700',
                  props.disabled && 'opacity-50'
                )}
              >
                {label}
                {props.required && <span className="text-red-500 ml-1">*</span>}
              </span>
              {helperText && !error && (
                <p className="mt-1 text-sm text-gray-500">{helperText}</p>
              )}
              {error && (
                <p className="mt-1 text-sm text-red-600" role="alert">
                  {error}
                </p>
              )}
            </div>
          )}
        </label>
      </div>
    )
  }
)

Checkbox.displayName = 'Checkbox'

export default Checkbox




