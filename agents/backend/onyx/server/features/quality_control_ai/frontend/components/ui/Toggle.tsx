'use client';

import { memo } from 'react';
import { useToggle } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Button } from './Button';

interface ToggleProps {
  value?: boolean;
  onToggle?: (value: boolean) => void;
  label?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
}

const Toggle = memo(
  ({
    value: controlledValue,
    onToggle,
    label,
    className,
    size = 'md',
    disabled = false,
  }: ToggleProps): JSX.Element => {
    const [internalValue, toggle, setToggle] = useToggle(controlledValue ?? false);
    const isControlled = controlledValue !== undefined;
    const value = isControlled ? controlledValue : internalValue;

    const handleToggle = (): void => {
      if (disabled) return;

      const newValue = !value;
      if (!isControlled) {
        setToggle(newValue);
      }
      onToggle?.(newValue);
    };

    const sizeClasses = {
      sm: 'w-8 h-4',
      md: 'w-11 h-6',
      lg: 'w-14 h-7',
    };

    const dotSizeClasses = {
      sm: 'w-3 h-3',
      md: 'w-5 h-5',
      lg: 'w-6 h-6',
    };

    const translateClasses = {
      sm: value ? 'translate-x-4' : 'translate-x-0',
      md: value ? 'translate-x-5' : 'translate-x-0',
      lg: value ? 'translate-x-7' : 'translate-x-0',
    };

    return (
      <div className={cn('flex items-center gap-2', className)}>
        {label && (
          <label
            className="text-sm font-medium text-gray-700 cursor-pointer"
            onClick={handleToggle}
          >
            {label}
          </label>
        )}
        <button
          type="button"
          role="switch"
          aria-checked={value}
          onClick={handleToggle}
          disabled={disabled}
          className={cn(
            'relative inline-flex items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500',
            sizeClasses[size],
            value ? 'bg-primary-600' : 'bg-gray-300',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <span
            className={cn(
              'inline-block rounded-full bg-white transform transition-transform',
              dotSizeClasses[size],
              translateClasses[size]
            )}
          />
        </button>
      </div>
    );
  }
);

Toggle.displayName = 'Toggle';

export default Toggle;

