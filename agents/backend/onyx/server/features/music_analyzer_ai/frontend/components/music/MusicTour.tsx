'use client';

import { useState, useEffect } from 'react';
import { Map, X, ArrowRight } from 'lucide-react';

interface TourStep {
  target: string;
  title: string;
  content: string;
  position: 'top' | 'bottom' | 'left' | 'right';
}

export function MusicTour() {
  const [isActive, setIsActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const steps: TourStep[] = [
    {
      target: 'search-tab',
      title: 'Búsqueda',
      content: 'Busca canciones, artistas o álbumes aquí',
      position: 'bottom',
    },
    {
      target: 'analysis-tab',
      title: 'Análisis',
      content: 'Analiza canciones para obtener insights detallados',
      position: 'bottom',
    },
    {
      target: 'compare-tab',
      title: 'Comparación',
      content: 'Compara múltiples canciones lado a lado',
      position: 'bottom',
    },
  ];

  useEffect(() => {
    if (isActive && steps[currentStep]) {
      const element = document.querySelector(`[data-tour="${steps[currentStep].target}"]`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [isActive, currentStep]);

  if (!isActive) {
    return (
      <button
        onClick={() => setIsActive(true)}
        className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
        title="Tour guiado"
      >
        <Map className="w-5 h-5" />
      </button>
    );
  }

  const currentStepData = steps[currentStep];
  if (!currentStepData) {
    setIsActive(false);
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 pointer-events-none">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
        <div
          className="bg-gradient-to-br from-purple-900 to-pink-900 rounded-xl p-6 max-w-md w-full border border-white/20 shadow-2xl pointer-events-auto"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">{currentStepData.title}</h3>
            <button
              onClick={() => setIsActive(false)}
              className="p-1 hover:bg-white/10 rounded transition-colors"
            >
              <X className="w-4 h-4 text-white" />
            </button>
          </div>
          <p className="text-gray-300 mb-4">{currentStepData.content}</p>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">
              {currentStep + 1} / {steps.length}
            </span>
            <button
              onClick={() => {
                if (currentStep < steps.length - 1) {
                  setCurrentStep(currentStep + 1);
                } else {
                  setIsActive(false);
                }
              }}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              {currentStep < steps.length - 1 ? (
                <>
                  Siguiente
                  <ArrowRight className="w-4 h-4" />
                </>
              ) : (
                'Finalizar'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}


