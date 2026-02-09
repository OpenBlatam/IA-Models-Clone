'use client';

import React, { useEffect, useRef } from 'react';
import { clsx } from 'clsx';

interface QRCodeProps {
  value: string;
  size?: number;
  level?: 'L' | 'M' | 'Q' | 'H';
  className?: string;
}

export const QRCode: React.FC<QRCodeProps> = ({
  value,
  size = 200,
  level = 'M',
  className,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Simple QR code generation (basic implementation)
    // In production, you'd use a library like qrcode.js
    const cellSize = 10;
    const margin = 4;
    const qrSize = size - margin * 2;
    const modules = Math.floor(qrSize / cellSize);

    canvas.width = size;
    canvas.height = size;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, size, size);

    ctx.fillStyle = '#000000';

    // Generate a simple pattern (this is a placeholder)
    // In production, use a proper QR code library
    for (let y = 0; y < modules; y++) {
      for (let x = 0; x < modules; x++) {
        const hash = (x + y * modules + value.length) % 3;
        if (hash === 0) {
          ctx.fillRect(
            margin + x * cellSize,
            margin + y * cellSize,
            cellSize,
            cellSize
          );
        }
      }
    }

    // Add finder patterns (corners)
    const finderSize = 7;
    const finderPositions = [
      [margin, margin],
      [size - margin - finderSize * cellSize, margin],
      [margin, size - margin - finderSize * cellSize],
    ];

    finderPositions.forEach(([x, y]) => {
      ctx.fillRect(x, y, finderSize * cellSize, finderSize * cellSize);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x + cellSize, y + cellSize, 5 * cellSize, 5 * cellSize);
      ctx.fillStyle = '#000000';
      ctx.fillRect(x + 2 * cellSize, y + 2 * cellSize, 3 * cellSize, 3 * cellSize);
    });
  }, [value, size, level]);

  return (
    <canvas
      ref={canvasRef}
      className={clsx('border border-gray-200 dark:border-gray-700 rounded-lg', className)}
      style={{ width: size, height: size }}
    />
  );
};


