import { useState, useCallback } from 'react';
import Input from './Input';
import { cn } from '@/lib/utils';

interface DatePickerProps {
  value?: string;
  onChange: (date: string) => void;
  label?: string;
  min?: string;
  max?: string;
  className?: string;
  required?: boolean;
}

const DatePicker = ({
  value,
  onChange,
  label,
  min,
  max,
  className = '',
  required = false,
}: DatePickerProps): JSX.Element => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>): void => {
      onChange(e.target.value);
    },
    [onChange]
  );

  return (
    <div className={cn('space-y-1', className)}>
      {label && (
        <label className="block text-sm font-medium text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <input
        type="date"
        value={value || ''}
        onChange={handleChange}
        min={min}
        max={max}
        required={required}
        className={cn(
          'w-full px-4 py-2 border border-gray-300 rounded-lg',
          'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent'
        )}
        aria-label={label || 'Date picker'}
      />
    </div>
  );
};

export default DatePicker;



