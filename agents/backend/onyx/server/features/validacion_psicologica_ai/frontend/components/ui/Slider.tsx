/**
 * Slider component
 */

'use client';

import React, { useRef, useState, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';

export interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  label?: string;
  showValue?: boolean;
}

export const Slider: React.FC<SliderProps> = ({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className,
  label,
  showValue = false,
}) => {
  const sliderRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const percentage = ((value - min) / (max - min)) * 100;

  const handleMouseDown = (event: React.MouseEvent) => {
    if (disabled) {
      return;
    }
    setIsDragging(true);
    updateValue(event);
  };

  const handleMouseMove = (event: MouseEvent) => {
    if (!isDragging || disabled) {
      return;
    }
    updateValue(event);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const updateValue = (event: MouseEvent | React.MouseEvent) => {
    if (!sliderRef.current) {
      return;
    }

    const rect = sliderRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const newValue = Math.round((min + percentage * (max - min)) / step) * step;
    const clampedValue = Math.max(min, Math.min(max, newValue));

    if (clampedValue !== value) {
      onChange(clampedValue);
    }
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (disabled) {
      return;
    }

    let newValue = value;

    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        event.preventDefault();
        newValue = Math.min(max, value + step);
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        event.preventDefault();
        newValue = Math.max(min, value - step);
        break;
      case 'Home':
        event.preventDefault();
        newValue = min;
        break;
      case 'End':
        event.preventDefault();
        newValue = max;
        break;
      default:
        return;
    }

    if (newValue !== value) {
      onChange(newValue);
    }
  };

  return (
    <div className={cn('w-full', className)}>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-2">
          {label && <label className="text-sm font-medium">{label}</label>}
          {showValue && <span className="text-sm text-muted-foreground">{value}</span>}
        </div>
      )}
      <div
        ref={sliderRef}
        className={cn(
          'relative h-2 w-full rounded-full bg-muted cursor-pointer',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
        onMouseDown={handleMouseDown}
        role="slider"
        aria-valuenow={value}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-disabled={disabled}
        aria-label={label || 'Slider'}
        tabIndex={disabled ? -1 : 0}
        onKeyDown={handleKeyDown}
      >
        <div
          className="absolute h-full rounded-full bg-primary"
          style={{ width: `${percentage}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-4 w-4 rounded-full bg-primary border-2 border-background shadow-md hover:scale-110 transition-transform"
          style={{ left: `${percentage}%` }}
        />
      </div>
    </div>
  );
};



