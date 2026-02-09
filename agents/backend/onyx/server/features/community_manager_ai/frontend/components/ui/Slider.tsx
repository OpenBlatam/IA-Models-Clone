'use client';

import * as SliderPrimitive from '@radix-ui/react-slider';
import { cn } from '@/lib/utils';

interface SliderProps {
  value?: number[];
  defaultValue?: number[];
  onValueChange?: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  showValue?: boolean;
}

export const Slider = ({
  value,
  defaultValue,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className,
  showValue = false,
}: SliderProps) => {
  const currentValue = value || defaultValue || [min];

  return (
    <div className={cn('w-full', className)}>
      {showValue && (
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">Valor</span>
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {currentValue[0]}
          </span>
        </div>
      )}
      <SliderPrimitive.Root
        value={value}
        defaultValue={defaultValue}
        onValueChange={onValueChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={cn(
          'relative flex w-full touch-none select-none items-center',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <SliderPrimitive.Track className="relative h-2 w-full grow rounded-full bg-gray-200 dark:bg-gray-700">
          <SliderPrimitive.Range className="absolute h-full rounded-full bg-primary-600 dark:bg-primary-500" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className={cn(
            'block h-5 w-5 rounded-full border-2 border-primary-600 dark:border-primary-500',
            'bg-white dark:bg-gray-800 shadow-md',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            'disabled:pointer-events-none disabled:opacity-50'
          )}
          aria-label="Value"
        />
      </SliderPrimitive.Root>
    </div>
  );
};



