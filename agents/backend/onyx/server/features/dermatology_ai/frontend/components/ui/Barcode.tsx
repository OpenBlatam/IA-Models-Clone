'use client';

import React, { useEffect, useRef } from 'react';
import { clsx } from 'clsx';

interface BarcodeProps {
  value: string;
  format?: 'CODE128' | 'CODE39' | 'EAN13';
  width?: number;
  height?: number;
  className?: string;
}

export const Barcode: React.FC<BarcodeProps> = ({
  value,
  format = 'CODE128',
  width = 2,
  height = 100,
  className,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Simple barcode generation (basic implementation)
    // In production, you'd use a library like JsBarcode
    const barWidth = width;
    let x = 0;

    canvas.width = value.length * barWidth * 10;
    canvas.height = height;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#000000';

    // Generate bars based on value
    value.split('').forEach((char, index) => {
      const code = char.charCodeAt(0);
      const pattern = code.toString(2).padStart(8, '0');

      pattern.split('').forEach((bit) => {
        if (bit === '1') {
          ctx.fillRect(x, 0, barWidth, height);
        }
        x += barWidth;
      });
    });
  }, [value, format, width, height]);

  return (
    <canvas
      ref={canvasRef}
      className={clsx('border border-gray-200 dark:border-gray-700 rounded', className)}
    />
  );
};


