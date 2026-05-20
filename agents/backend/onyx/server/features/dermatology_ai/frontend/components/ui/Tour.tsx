'use client';

import React, { useState, useEffect, useRef } from 'react';
import { X, ChevronLeft, ChevronRight } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from './Button';
import { Portal } from './Portal';

interface TourStep {
  target: string;
  title: string;
  content: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
}

interface TourProps {
  steps: TourStep[];
  isOpen: boolean;
  onClose: () => void;
  onComplete?: () => void;
}

export const Tour: React.FC<TourProps> = ({
  steps,
  isOpen,
  onClose,
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen || steps.length === 0) return;

    const current = steps[currentStep];
    const targetElement = document.querySelector(current.target);

    if (targetElement) {
      const rect = targetElement.getBoundingClientRect();
      const placement = current.placement || 'bottom';

      let top = 0;
      let left = 0;

      switch (placement) {
        case 'top':
          top = rect.top - 20;
          left = rect.left + rect.width / 2;
          break;
        case 'bottom':
          top = rect.bottom + 20;
          left = rect.left + rect.width / 2;
          break;
        case 'left':
          top = rect.top + rect.height / 2;
          left = rect.left - 20;
          break;
        case 'right':
          top = rect.top + rect.height / 2;
          left = rect.right + 20;
          break;
      }

      setPosition({ top, left });
      targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [isOpen, currentStep, steps]);

  if (!isOpen || steps.length === 0) return null;

  const current = steps[currentStep];
  const isFirst = currentStep === 0;
  const isLast = currentStep === steps.length - 1;

  const handleNext = () => {
    if (isLast) {
      onComplete?.();
      onClose();
    } else {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (!isFirst) {
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <Portal>
      <div className="fixed inset-0 z-50">
        {/* Overlay */}
        <div
          ref={overlayRef}
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={onClose}
        />
        {/* Highlight */}
        <div
          className="absolute border-4 border-primary-500 rounded-lg pointer-events-none z-10"
          style={{
            top: position.top - 100,
            left: position.left - 100,
            width: '200px',
            height: '200px',
          }}
        />
        {/* Tooltip */}
        <div
          className="absolute bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-sm z-20"
          style={{
            top: `${position.top}px`,
            left: `${position.left}px`,
            transform: 'translate(-50%, 0)',
          }}
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">
                {current.title}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Paso {currentStep + 1} de {steps.length}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="mb-4 text-gray-700 dark:text-gray-300">
            {current.content}
          </div>
          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevious}
              disabled={isFirst}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Anterior
            </Button>
            <Button size="sm" onClick={handleNext}>
              {isLast ? 'Completar' : 'Siguiente'}
              {!isLast && <ChevronRight className="h-4 w-4 ml-1" />}
            </Button>
          </div>
        </div>
      </div>
    </Portal>
  );
};


