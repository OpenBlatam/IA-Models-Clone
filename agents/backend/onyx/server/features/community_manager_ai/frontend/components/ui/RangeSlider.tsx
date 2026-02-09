'use client';

import * as SliderPrimitive from '@radix-ui/react-slider';
import { cn } from '@/lib/utils';

interface RangeSliderProps {
  value?: number[];
  defaultValue?: number[];
  onValueChange?: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  showValues?: boolean;
  label?: string;
}

export const RangeSlider = ({
  value,
  defaultValue,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className,
  showValues = true,
  label,
}: RangeSliderProps) => {
  const currentValue = value || defaultValue || [min, max];

  return (
    <div className={cn('w-full space-y-2', className)}>
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
          {showValues && (
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {currentValue[0]} - {currentValue[1]}
            </span>
          )}
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
          aria-label="Minimum value"
        />
        <SliderPrimitive.Thumb
          className={cn(
            'block h-5 w-5 rounded-full border-2 border-primary-600 dark:border-primary-500',
            'bg-white dark:bg-gray-800 shadow-md',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            'disabled:pointer-events-none disabled:opacity-50'
          )}
          aria-label="Maximum value"
        />
      </SliderPrimitive.Root>
    </div>
  );
};



