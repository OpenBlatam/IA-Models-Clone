'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';
import { Label } from './Label';
import { Input } from './Input';

interface PriceFilterProps {
  minPrice?: number;
  maxPrice?: number;
  onPriceChange: (min: number, max: number) => void;
  className?: string;
}

export default function PriceFilter({
  minPrice = 0,
  maxPrice = 10000,
  onPriceChange,
  className,
}: PriceFilterProps) {
  const [min, setMin] = useState(minPrice);
  const [max, setMax] = useState(maxPrice);

  const handleMinChange = (value: number) => {
    const newMin = Math.max(0, Math.min(value, max));
    setMin(newMin);
    onPriceChange(newMin, max);
  };

  const handleMaxChange = (value: number) => {
    const newMax = Math.max(min, Math.min(value, 100000));
    setMax(newMax);
    onPriceChange(min, newMax);
  };

  return (
    <div className={cn('space-y-4', className)}>
      <div>
        <Label className="mb-2">Rango de Precio</Label>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <Label htmlFor="min-price" className="text-xs text-tesla-gray-dark mb-1">
              Mínimo
            </Label>
            <Input
              id="min-price"
              type="number"
              value={min}
              onChange={(e) => handleMinChange(Number(e.target.value))}
              min={0}
              max={max}
              className="w-full"
            />
          </div>
          <div className="flex-1">
            <Label htmlFor="max-price" className="text-xs text-tesla-gray-dark mb-1">
              Máximo
            </Label>
            <Input
              id="max-price"
              type="number"
              value={max}
              onChange={(e) => handleMaxChange(Number(e.target.value))}
              min={min}
              max={100000}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Range Slider */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-tesla-gray-dark">
          <span>${min.toLocaleString()}</span>
          <span>${max.toLocaleString()}</span>
        </div>
        <div className="relative">
          <input
            type="range"
            min={minPrice}
            max={maxPrice}
            value={min}
            onChange={(e) => handleMinChange(Number(e.target.value))}
            className="absolute w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-tesla-blue"
            style={{
              zIndex: min > max - 100 ? 2 : 1,
            }}
          />
          <input
            type="range"
            min={minPrice}
            max={maxPrice}
            value={max}
            onChange={(e) => handleMaxChange(Number(e.target.value))}
            className="absolute w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-tesla-blue"
          />
        </div>
      </div>
    </div>
  );
}



