'use client';

import { useState } from 'react';
import { BookOpen, X, ChevronRight, ChevronLeft } from 'lucide-react';

interface TutorialStep {
  title: string;
  content: string;
  target?: string;
}

export function MusicTutorial() {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const steps: TutorialStep[] = [
    {
      title: 'Bienvenido a Music Analyzer AI',
      content: 'Explora y analiza música con inteligencia artificial. Comienza buscando una canción.',
    },
    {
      title: 'Búsqueda Avanzada',
      content: 'Usa filtros, ordenamiento y vistas para encontrar exactamente lo que buscas.',
    },
    {
      title: 'Análisis Detallado',
      content: 'Obtén insights completos sobre cualquier canción: energía, danceability, tempo y más.',
    },
    {
      title: 'Comparación de Canciones',
      content: 'Compara múltiples canciones lado a lado para encontrar similitudes y diferencias.',
    },
    {
      title: 'Recomendaciones Inteligentes',
      content: 'Recibe recomendaciones personalizadas basadas en tu música favorita.',
    },
    {
      title: 'Gestión de Playlists',
      content: 'Crea y gestiona playlists inteligentes basadas en mood, actividad o género.',
    },
  ];

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 p-4 bg-purple-600 hover:bg-purple-700 text-white rounded-full shadow-lg transition-colors z-50"
        title="Tutorial"
      >
        <BookOpen className="w-6 h-6" />
      </button>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-purple-900 to-pink-900 rounded-xl p-6 max-w-2xl w-full border border-white/20 shadow-2xl">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Tutorial</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        </div>

        <div className="mb-6">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm text-gray-400">
              Paso {currentStep + 1} de {steps.length}
            </span>
            <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 transition-all"
                style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
              />
            </div>
          </div>

          <h3 className="text-xl font-semibold text-white mb-2">
            {steps[currentStep].title}
          </h3>
          <p className="text-gray-300">{steps[currentStep].content}</p>
        </div>

        <div className="flex items-center justify-between">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            <ChevronLeft className="w-4 h-4" />
            Anterior
          </button>
          <div className="flex gap-2">
            {steps.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentStep(idx)}
                className={`w-2 h-2 rounded-full transition-colors ${
                  idx === currentStep ? 'bg-purple-500' : 'bg-white/30'
                }`}
              />
            ))}
          </div>
          <button
            onClick={() => {
              if (currentStep < steps.length - 1) {
                setCurrentStep(currentStep + 1);
              } else {
                setIsOpen(false);
              }
            }}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            {currentStep < steps.length - 1 ? (
              <>
                Siguiente
                <ChevronRight className="w-4 h-4" />
              </>
            ) : (
              'Finalizar'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}


