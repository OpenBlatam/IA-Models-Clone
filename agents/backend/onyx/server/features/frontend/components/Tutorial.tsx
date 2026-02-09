'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiChevronRight, FiChevronLeft, FiCheck } from 'react-icons/fi';

interface TutorialStep {
  id: string;
  title: string;
  description: string;
  target?: string; // CSS selector
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center';
}

const tutorialSteps: TutorialStep[] = [
  {
    id: 'welcome',
    title: '¡Bienvenido a BUL!',
    description: 'Esta es una guía rápida para ayudarte a comenzar. Puedes saltarla o seguirla paso a paso.',
    position: 'center',
  },
  {
    id: 'dashboard',
    title: 'Dashboard',
    description: 'Aquí puedes ver un resumen de todas tus tareas y métricas del sistema.',
    target: '[data-tutorial="dashboard"]',
    position: 'bottom',
  },
  {
    id: 'generate',
    title: 'Generar Documentos',
    description: 'Usa esta vista para crear nuevos documentos con IA. Puedes usar plantillas, voz o escribir manualmente.',
    target: '[data-tutorial="generate"]',
    position: 'bottom',
  },
  {
    id: 'tasks',
    title: 'Gestionar Tareas',
    description: 'Revisa el estado de todas tus tareas, filtra por estado, fecha o prioridad.',
    target: '[data-tutorial="tasks"]',
    position: 'bottom',
  },
  {
    id: 'shortcuts',
    title: 'Atajos de Teclado',
    description: 'Presiona Ctrl+/ para ver todos los atajos disponibles. Usa Ctrl+K para búsqueda global.',
    position: 'center',
  },
];

interface TutorialProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Tutorial({ isOpen, onClose }: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    const step = tutorialSteps[currentStep];
    if (step?.target) {
      const element = document.querySelector(step.target) as HTMLElement;
      setTargetElement(element);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        element.style.zIndex = '1000';
        element.style.position = 'relative';
      }
    } else {
      setTargetElement(null);
    }

    return () => {
      if (element) {
        element.style.zIndex = '';
        element.style.position = '';
      }
    };
  }, [currentStep, isOpen]);

  const handleNext = () => {
    if (currentStep < tutorialSteps.length - 1) {
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

  const handleComplete = () => {
    localStorage.setItem('bul_tutorial_completed', 'true');
    onClose();
  };

  const handleSkip = () => {
    localStorage.setItem('bul_tutorial_completed', 'true');
    onClose();
  };

  if (!isOpen) return null;

  const step = tutorialSteps[currentStep];
  const isLast = currentStep === tutorialSteps.length - 1;
  const isFirst = currentStep === 0;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50"
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-black bg-opacity-50" onClick={handleSkip} />

        {/* Tutorial Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className={`absolute ${
            step.position === 'center'
              ? 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2'
              : step.position === 'top'
              ? 'top-4 left-1/2 transform -translate-x-1/2'
              : step.position === 'bottom'
              ? 'bottom-4 left-1/2 transform -translate-x-1/2'
              : step.position === 'left'
              ? 'left-4 top-1/2 transform -translate-y-1/2'
              : 'right-4 top-1/2 transform -translate-y-1/2'
          } bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 max-w-md w-full mx-4`}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-primary-600">
                {currentStep + 1} / {tutorialSteps.length}
              </span>
            </div>
            <button onClick={handleSkip} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            {step.title}
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">{step.description}</p>

          <div className="flex items-center justify-between">
            <button
              onClick={handlePrevious}
              disabled={isFirst}
              className="btn btn-secondary"
            >
              <FiChevronLeft size={18} className="mr-1" />
              Anterior
            </button>
            <div className="flex gap-2">
              {tutorialSteps.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full ${
                    index === currentStep
                      ? 'bg-primary-600'
                      : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                />
              ))}
            </div>
            <button
              onClick={isLast ? handleComplete : handleNext}
              className="btn btn-primary"
            >
              {isLast ? (
                <>
                  <FiCheck size={18} className="mr-1" />
                  Completar
                </>
              ) : (
                <>
                  Siguiente
                  <FiChevronRight size={18} className="ml-1" />
                </>
              )}
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


