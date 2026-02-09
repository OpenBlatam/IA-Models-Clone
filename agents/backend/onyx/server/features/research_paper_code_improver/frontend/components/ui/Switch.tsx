import React from 'react'
import { clsx } from 'clsx'

interface SwitchProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  helperText?: string
}

const Switch = React.forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, label, helperText, ...props }, ref) => {
    return (
      <div className="w-full">
        <label className="flex items-center gap-3 cursor-pointer group">
          <div className="relative flex items-center">
            <input
              ref={ref}
              type="checkbox"
              className="sr-only"
              {...props}
            />
            <div
              className={clsx(
                'w-11 h-6 rounded-full transition-colors relative',
                props.checked
                  ? 'bg-primary-600'
                  : 'bg-gray-300',
                props.disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <div
                className={clsx(
                  'absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform shadow-sm',
                  props.checked ? 'translate-x-5' : 'translate-x-0'
                )}
              />
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
              {helperText && (
                <p className="text-xs text-gray-500 mt-1">{helperText}</p>
              )}
            </div>
          )}
        </label>
      </div>
    )
  }
)

Switch.displayName = 'Switch'

export default Switch




