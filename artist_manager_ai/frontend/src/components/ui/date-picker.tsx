'use client';

import { forwardRef } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { cn } from '@/lib/utils';
import { Calendar } from 'lucide-react';

interface DatePickerProps {
  label?: string;
  error?: string;
  value?: Date | null;
  onChange: (date: Date | null) => void;
  showTimeSelect?: boolean;
  dateFormat?: string;
  minDate?: Date;
  maxDate?: Date;
  className?: string;
  required?: boolean;
  placeholder?: string;
}

const CustomDatePicker = forwardRef<HTMLInputElement, DatePickerProps>(
  (
    {
      label,
      error,
      value,
      onChange,
      showTimeSelect = false,
      dateFormat,
      minDate,
      maxDate,
      className,
      required,
      placeholder,
    },
    ref
  ) => {
    const defaultDateFormat = showTimeSelect ? 'dd/MM/yyyy HH:mm' : 'dd/MM/yyyy';

    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}
        <div className="relative">
          <DatePicker
            ref={ref}
            selected={value}
            onChange={onChange}
            showTimeSelect={showTimeSelect}
            timeIntervals={15}
            dateFormat={dateFormat || defaultDateFormat}
            minDate={minDate}
            maxDate={maxDate}
            placeholderText={placeholder}
            className={cn(
              'w-full px-3 py-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
              error && 'border-red-500 focus:ring-red-500',
              className
            )}
            required={required}
            aria-invalid={error ? 'true' : 'false'}
            aria-describedby={error ? `${label}-error` : undefined}
          />
          <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
        {error && (
          <p id={`${label}-error`} className="mt-1 text-sm text-red-600" role="alert">
            {error}
          </p>
        )}
      </div>
    );
  }
);

CustomDatePicker.displayName = 'CustomDatePicker';

export { CustomDatePicker };

