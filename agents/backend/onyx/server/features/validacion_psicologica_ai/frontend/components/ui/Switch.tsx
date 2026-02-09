/**
 * Switch toggle component with accessibility
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface SwitchProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  description?: string;
}

const Switch = React.forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, label, description, id, checked, onChange, disabled, ...props }, ref) => {
    const switchId = id || `switch-${Math.random().toString(36).substr(2, 9)}`;

    const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        if (!disabled && onChange) {
          onChange({
            ...event,
            target: { ...event.target, checked: !checked } as HTMLInputElement,
          } as React.ChangeEvent<HTMLInputElement>);
        }
      }
    };

    return (
      <div className="flex items-start space-x-3">
        <div className="flex items-center h-5">
          <input
            ref={ref}
            type="checkbox"
            id={switchId}
            checked={checked}
            onChange={onChange}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            className={cn(
              'sr-only peer',
              className
            )}
            role="switch"
            aria-checked={checked}
            {...props}
          />
          <label
            htmlFor={switchId}
            className={cn(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer',
              'peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2',
              checked ? 'bg-primary' : 'bg-input',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
            aria-hidden="true"
          >
            <span
              className={cn(
                'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                checked ? 'translate-x-6' : 'translate-x-1'
              )}
            />
          </label>
        </div>
        {(label || description) && (
          <div className="flex-1">
            {label && (
              <label
                htmlFor={switchId}
                className={cn(
                  'text-sm font-medium cursor-pointer',
                  disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                {label}
              </label>
            )}
            {description && (
              <p className="text-xs text-muted-foreground mt-1">{description}</p>
            )}
          </div>
        )}
      </div>
    );
  }
);

Switch.displayName = 'Switch';

export { Switch };




