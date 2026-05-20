'use client';

import * as RadioGroupPrimitive from '@radix-ui/react-radio-group';
import { Circle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface RadioGroupProps {
  value?: string;
  onValueChange?: (value: string) => void;
  children: React.ReactNode;
  className?: string;
}

interface RadioItemProps {
  value: string;
  label: string;
  className?: string;
}

const RadioGroup = ({ value, onValueChange, children, className }: RadioGroupProps) => {
  return (
    <RadioGroupPrimitive.Root
      value={value}
      onValueChange={onValueChange}
      className={cn('grid gap-2', className)}
    >
      {children}
    </RadioGroupPrimitive.Root>
  );
};

const RadioItem = ({ value, label, className }: RadioItemProps) => {
  return (
    <div className="flex items-center gap-2">
      <RadioGroupPrimitive.Item
        value={value}
        className={cn(
          'aspect-square h-4 w-4 rounded-full border border-gray-300 text-blue-600 ring-offset-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
      >
        <RadioGroupPrimitive.Indicator className="flex items-center justify-center">
          <Circle className="h-2.5 w-2.5 fill-current text-current" />
        </RadioGroupPrimitive.Indicator>
      </RadioGroupPrimitive.Item>
      <label className="text-sm font-medium text-gray-700 cursor-pointer">{label}</label>
    </div>
  );
};

export { RadioGroup, RadioItem };

