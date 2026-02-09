'use client';

import * as RadioGroupPrimitive from '@radix-ui/react-radio-group';
import { Circle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface RadioOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

interface RadioGroupProps {
  options: RadioOption[];
  value?: string;
  onValueChange?: (value: string) => void;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

export const RadioGroup = ({
  options,
  value,
  onValueChange,
  className,
  orientation = 'vertical',
}: RadioGroupProps) => {
  return (
    <RadioGroupPrimitive.Root
      value={value}
      onValueChange={onValueChange}
      className={cn(
        'space-y-2',
        orientation === 'horizontal' && 'flex flex-wrap gap-4',
        className
      )}
    >
      {options.map((option) => (
        <div key={option.value} className="flex items-start gap-2">
          <RadioGroupPrimitive.Item
            value={option.value}
            disabled={option.disabled}
            className={cn(
              'mt-0.5 h-4 w-4 rounded-full border-2 border-gray-300 dark:border-gray-700',
              'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
              'disabled:cursor-not-allowed disabled:opacity-50',
              'data-[state=checked]:border-primary-600 dark:data-[state=checked]:border-primary-500'
            )}
            id={option.value}
          >
            <RadioGroupPrimitive.Indicator className="flex items-center justify-center">
              <Circle className="h-2.5 w-2.5 fill-primary-600 dark:fill-primary-500 text-primary-600 dark:text-primary-500" />
            </RadioGroupPrimitive.Indicator>
          </RadioGroupPrimitive.Item>
          <label
            htmlFor={option.value}
            className={cn(
              'text-sm font-medium cursor-pointer',
              option.disabled && 'opacity-50 cursor-not-allowed',
              'text-gray-900 dark:text-gray-100'
            )}
          >
            {option.label}
            {option.description && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                {option.description}
              </p>
            )}
          </label>
        </div>
      ))}
    </RadioGroupPrimitive.Root>
  );
};



