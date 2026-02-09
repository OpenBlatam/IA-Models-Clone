/**
 * Date picker component
 */

import React from 'react';
import { Input } from './Input';
import { Calendar } from 'lucide-react';

export interface DatePickerProps {
  value?: string;
  onChange: (value: string) => void;
  label?: string;
  min?: string;
  max?: string;
  error?: string;
  className?: string;
}

const DatePicker: React.FC<DatePickerProps> = ({
  value,
  onChange,
  label,
  min,
  max,
  error,
  className,
}) => {
  const inputId = `date-picker-${Math.random().toString(36).substr(2, 9)}`;

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  return (
    <div className={className}>
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium mb-1 text-foreground">
          {label}
        </label>
      )}
      <div className="relative">
        <Input
          id={inputId}
          type="date"
          value={value}
          onChange={handleChange}
          min={min}
          max={max}
          error={error}
          className="pl-10"
        />
        <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" aria-hidden="true" />
      </div>
    </div>
  );
};

export { DatePicker };




