'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from './Button';

interface CarouselProps {
  children: React.ReactNode[];
  autoPlay?: boolean;
  interval?: number;
  showControls?: boolean;
  showIndicators?: boolean;
  className?: string;
}

export const Carousel: React.FC<CarouselProps> = ({
  children,
  autoPlay = false,
  interval = 3000,
  showControls = true,
  showIndicators = true,
  className,
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (autoPlay) {
      intervalRef.current = setInterval(() => {
        setCurrentIndex((prev) => (prev + 1) % children.length);
      }, interval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoPlay, interval, children.length]);

  const goToSlide = (index: number) => {
    setCurrentIndex(index);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  const goToPrevious = () => {
    setCurrentIndex((prev) => (prev - 1 + children.length) % children.length);
  };

  const goToNext = () => {
    setCurrentIndex((prev) => (prev + 1) % children.length);
  };

  return (
    <div className={clsx('relative w-full', className)}>
      <div className="overflow-hidden rounded-lg">
        <div
          className="flex transition-transform duration-500 ease-in-out"
          style={{ transform: `translateX(-${currentIndex * 100}%)` }}
        >
          {children.map((child, index) => (
            <div key={index} className="min-w-full">
              {child}
            </div>
          ))}
        </div>
      </div>

      {showControls && (
        <>
          <button
            onClick={goToPrevious}
            className="absolute left-4 top-1/2 -translate-y-1/2 p-2 bg-white/80 dark:bg-gray-800/80 rounded-full shadow-lg hover:bg-white dark:hover:bg-gray-800 transition-colors"
            aria-label="Anterior"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
          <button
            onClick={goToNext}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-2 bg-white/80 dark:bg-gray-800/80 rounded-full shadow-lg hover:bg-white dark:hover:bg-gray-800 transition-colors"
            aria-label="Siguiente"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        </>
      )}

      {showIndicators && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex space-x-2">
          {children.map((_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={clsx(
                'h-2 rounded-full transition-all',
                index === currentIndex
                  ? 'w-8 bg-primary-600 dark:bg-primary-400'
                  : 'w-2 bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500'
              )}
              aria-label={`Ir a slide ${index + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  );
};


