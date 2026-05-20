'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiZap, FiShield, FiRocket, FiCheck } from 'react-icons/fi';

interface WelcomeScreenProps {
  isOpen: boolean;
  onClose: () => void;
}

const features = [
  {
    icon: FiZap,
    title: 'Generación Rápida',
    description: 'Crea documentos profesionales en minutos usando IA',
  },
  {
    icon: FiShield,
    title: 'Seguro y Privado',
    description: 'Tus datos están protegidos y seguros',
  },
  {
    icon: FiRocket,
    title: 'Potente y Flexible',
    description: 'Múltiples formatos, plantillas y opciones de exportación',
  },
];

export default function WelcomeScreen({ isOpen, onClose }: WelcomeScreenProps) {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (isOpen) {
      const timer = setInterval(() => {
        setCurrentStep((prev) => (prev < features.length - 1 ? prev + 1 : 0));
      }, 3000);
      return () => clearInterval(timer);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
                ¡Bienvenido a BUL!
              </h2>
              <button onClick={onClose} className="btn-icon">
                <FiX size={24} />
              </button>
            </div>

            <p className="text-gray-600 dark:text-gray-400 mb-8">
              La plataforma más avanzada para generar documentos de negocio con inteligencia artificial.
            </p>

            <div className="space-y-4 mb-8">
              {features.map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex items-start gap-4 p-4 rounded-lg transition-colors ${
                    currentStep === index
                      ? 'bg-primary-50 dark:bg-primary-900/20 border-2 border-primary-500'
                      : 'bg-gray-50 dark:bg-gray-700'
                  }`}
                >
                  <feature.icon
                    size={24}
                    className={`flex-shrink-0 ${
                      currentStep === index ? 'text-primary-600' : 'text-gray-400'
                    }`}
                  />
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {feature.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {feature.description}
                    </p>
                  </div>
                  {currentStep === index && (
                    <FiCheck size={20} className="text-primary-600 ml-auto" />
                  )}
                </motion.div>
              ))}
            </div>

            <div className="flex items-center justify-between">
              <div className="flex gap-2">
                {features.map((_, index) => (
                  <div
                    key={index}
                    className={`w-2 h-2 rounded-full transition-colors ${
                      currentStep === index
                        ? 'bg-primary-600 w-8'
                        : 'bg-gray-300 dark:bg-gray-600'
                    }`}
                  />
                ))}
              </div>
              <button onClick={onClose} className="btn btn-primary">
                Comenzar
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


