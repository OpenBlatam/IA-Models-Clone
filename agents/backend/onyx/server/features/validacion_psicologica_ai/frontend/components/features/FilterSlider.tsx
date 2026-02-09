/**
 * Filter slider component for date range or numeric filters
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Slider } from '@/components/ui';
import { X } from 'lucide-react';
import { Button } from '@/components/ui';

export interface FilterSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  onClear?: () => void;
}

export const FilterSlider: React.FC<FilterSliderProps> = ({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  unit = '',
  onClear,
}) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{label}</CardTitle>
          {onClear && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onClear();
                }
              }}
              aria-label="Limpiar filtro"
              tabIndex={0}
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Slider
          value={value}
          onChange={onChange}
          min={min}
          max={max}
          step={step}
          label={label}
          showValue
        />
        <div className="mt-2 text-sm text-muted-foreground">
          Valor actual: {value} {unit}
        </div>
      </CardContent>
    </Card>
  );
};



