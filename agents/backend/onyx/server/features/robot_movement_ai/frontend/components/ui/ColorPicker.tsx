'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface Color {
  name: string;
  value: string;
  hex: string;
}

interface ColorPickerProps {
  colors: Color[];
  selectedColor?: string;
  onSelect: (color: Color) => void;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

const sizeClasses = {
  sm: 'w-8 h-8',
  md: 'w-12 h-12',
  lg: 'w-16 h-16',
};

export default function ColorPicker({
  colors,
  selectedColor,
  onSelect,
  className,
  size = 'md',
}: ColorPickerProps) {
  return (
    <div className={cn('flex flex-wrap gap-tesla-sm', className)}>
      {colors.map((color) => (
        <motion.button
          key={color.value}
          onClick={() => onSelect(color)}
          className={cn(
            'rounded-full border-2 transition-all relative overflow-hidden',
            sizeClasses[size],
            selectedColor === color.value
              ? 'border-tesla-blue ring-4 ring-tesla-blue/20 scale-110'
              : 'border-gray-300 hover:border-gray-400'
          )}
          style={{ backgroundColor: color.hex }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          aria-label={`Seleccionar color ${color.name}`}
        >
          {selectedColor === color.value && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute inset-0 flex items-center justify-center"
            >
              <Check className="w-5 h-5 text-white drop-shadow-lg" />
            </motion.div>
          )}
        </motion.button>
      ))}
    </div>
  );
}

