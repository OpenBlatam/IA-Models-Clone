'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { storage } from '@/lib/utils/localStorage';
import { X, ArrowRight, ArrowLeft, CheckCircle } from 'lucide-react';

interface TourStep {
  id: string;
  title: string;
  description: string;
  target?: string; // CSS selector
}

export default function OnboardingTour() {
  const [isVisible, setIsVisible] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Check localStorage on mount
  useEffect(() => {
    const completed = storage.get<boolean>('onboarding-completed', false);
    setIsVisible(!completed);
  }, []);

  const handleComplete = useCallback(() => {
    storage.set('onboarding-completed', true);
    setIsVisible(false);
  }, []);

  // Close on ESC key press
  useEffect(() => {
    if (!isVisible) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleComplete();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isVisible, handleComplete]);

  const steps: TourStep[] = [
    {
      id: 'welcome',
      title: '¡Bienvenido a Robot Movement AI!',
      description: 'Esta es una plataforma completa para controlar y monitorear robots. Te guiaremos por las características principales.',
    },
    {
      id: 'control',
      title: 'Control del Robot',
      description: 'Usa la pestaña de Control para mover el robot, detenerlo o llevarlo a la posición inicial.',
    },
    {
      id: '3d',
      title: 'Visualización 3D',
      description: 'La visualización 3D te permite ver el robot en tiempo real desde cualquier ángulo.',
    },
    {
      id: 'chat',
      title: 'Chat Inteligente',
      description: 'Puedes comunicarte con el robot usando comandos de texto natural.',
    },
    {
      id: 'metrics',
      title: 'Métricas y Análisis',
      description: 'Monitorea el rendimiento del robot con métricas en tiempo real y análisis avanzados.',
    },
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    handleComplete();
  };

  // Close tour when clicking on tabs or header
  useEffect(() => {
    if (!isVisible) return;

    const handleTabClick = () => {
      handleComplete();
    };

    // Listen for clicks on tab buttons
    const tabButtons = document.querySelectorAll('[role="tab"]');
    tabButtons.forEach((tab) => {
      tab.addEventListener('click', handleTabClick);
    });

    return () => {
      tabButtons.forEach((tab) => {
        tab.removeEventListener('click', handleTabClick);
      });
    };
  }, [isVisible, handleComplete]);

  if (!isVisible) return null;

  const currentStepData = steps[currentStep];

  return (
    <AnimatePresence>
      {isVisible && (
        <>
          {/* Backdrop that covers only content area, not header/tabs */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-30 bg-black/20 backdrop-blur-sm"
            style={{ 
              top: '200px' // Start below header and tabs area
            }}
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                handleComplete();
              }
            }}
          />
          {/* Modal positioned in center */}
          <div 
            className="fixed inset-0 z-40 flex items-center justify-center pointer-events-none"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="relative max-w-md w-full mx-4 bg-white rounded-lg border border-gray-200 shadow-tesla-lg pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close Button - More prominent */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleComplete();
                }}
                className="absolute -top-2 -right-2 p-2 bg-white border-2 border-gray-300 hover:border-red-500 text-tesla-black rounded-full transition-all duration-200 shadow-tesla-md z-10 min-h-[44px] min-w-[44px] flex items-center justify-center"
                title="Cerrar (ESC)"
                aria-label="Cerrar tour"
              >
                <X className="w-5 h-5" />
              </motion.button>

              {/* Content */}
              <div className="p-6">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={currentStep}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xl font-bold text-tesla-black">{currentStepData.title}</h3>
                        <span className="text-sm text-tesla-gray-dark font-medium">
                          {currentStep + 1} / {steps.length}
                        </span>
                      </div>
                      <p className="text-tesla-gray-dark">{currentStepData.description}</p>
                    </div>
                  </motion.div>
                </AnimatePresence>

                {/* Progress */}
                <div className="mb-6">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                      transition={{ duration: 0.3, ease: "easeOut" }}
                      className="h-full bg-tesla-blue rounded-full"
                    />
                  </div>
                </div>

                {/* Steps Indicators */}
                <div className="flex gap-2 mb-6 justify-center">
                  {steps.map((step, index) => (
                    <motion.div
                      key={step.id}
                      initial={{ scale: 0.8 }}
                      animate={{ 
                        scale: index <= currentStep ? 1.2 : 1,
                        backgroundColor: index <= currentStep ? '#0062cc' : '#e5e7eb'
                      }}
                      transition={{ duration: 0.2 }}
                      className="w-2 h-2 rounded-full"
                    />
                  ))}
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <AnimatePresence>
                    {currentStep > 0 && (
                      <motion.button
                        initial={{ opacity: 0, width: 0 }}
                        animate={{ opacity: 1, width: 'auto' }}
                        exit={{ opacity: 0, width: 0 }}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handlePrevious}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium min-h-[44px]"
                      >
                        <ArrowLeft className="w-4 h-4" />
                        Anterior
                      </motion.button>
                    )}
                  </AnimatePresence>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleNext}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-tesla-blue hover:bg-opacity-90 text-white rounded-md transition-all font-medium min-h-[44px]"
                  >
                    {currentStep === steps.length - 1 ? (
                      <>
                        <CheckCircle className="w-4 h-4" />
                        Completar
                      </>
                    ) : (
                      <>
                        Siguiente
                        <ArrowRight className="w-4 h-4" />
                      </>
                    )}
                  </motion.button>
                </div>

                {/* Skip */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleComplete}
                  className="w-full mt-3 text-sm text-tesla-gray-dark hover:text-tesla-blue transition-colors underline font-medium"
                  title="O presiona ESC para cerrar"
                >
                  Omitir tour
                </motion.button>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}


