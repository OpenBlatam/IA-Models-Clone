'use client';

import { memo, useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { Input } from './Input';
import { Label } from './Label';
import { isValidHex, hexToRgb, rgbToHex } from '@/lib/utils';

interface ColorPickerProps {
  value?: string;
  onChange?: (color: string) => void;
  label?: string;
  className?: string;
  showInput?: boolean;
}

const ColorPicker = memo(
  ({
    value = '#000000',
    onChange,
    label,
    className,
    showInput = true,
  }: ColorPickerProps): JSX.Element => {
    const [inputValue, setInputValue] = useState(value);

    const handleColorChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const newColor = e.target.value;
        setInputValue(newColor);
        onChange?.(newColor);
      },
      [onChange]
    );

    const handleInputChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        setInputValue(newValue);

        if (isValidHex(newValue)) {
          onChange?.(newValue);
        }
      },
      [onChange]
    );

    return (
      <div className={cn('space-y-2', className)}>
        {label && <Label>{label}</Label>}
        <div className="flex items-center gap-2">
          <div className="relative">
            <input
              type="color"
              value={value}
              onChange={handleColorChange}
              className="w-12 h-12 rounded border border-gray-300 cursor-pointer"
              aria-label="Color picker"
            />
          </div>
          {showInput && (
            <Input
              type="text"
              value={inputValue}
              onChange={handleInputChange}
              placeholder="#000000"
              className="flex-1 font-mono"
              maxLength={7}
            />
          )}
        </div>
      </div>
    );
  }
);

ColorPicker.displayName = 'ColorPicker';

export default ColorPicker;

