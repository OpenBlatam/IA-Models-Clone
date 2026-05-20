'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFlag, FiX, FiToggleLeft, FiToggleRight } from 'react-icons/fi';

interface FeatureFlag {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
}

const defaultFlags: FeatureFlag[] = [
  {
    id: 'beta-features',
    name: 'Características Beta',
    description: 'Habilita características experimentales',
    enabled: false,
  },
  {
    id: 'advanced-analytics',
    name: 'Analytics Avanzados',
    description: 'Muestra métricas detalladas adicionales',
    enabled: true,
  },
  {
    id: 'experimental-ui',
    name: 'UI Experimental',
    description: 'Interfaz experimental con nuevas animaciones',
    enabled: false,
  },
  {
    id: 'debug-mode',
    name: 'Modo Debug',
    description: 'Muestra información de depuración',
    enabled: false,
  },
];

interface FeatureFlagsProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function FeatureFlags({ isOpen, onClose }: FeatureFlagsProps) {
  const [flags, setFlags] = useState<FeatureFlag[]>(defaultFlags);

  useEffect(() => {
    if (isOpen) {
      const stored = localStorage.getItem('bul_feature_flags');
      if (stored) {
        const storedFlags = JSON.parse(stored);
        setFlags(
          defaultFlags.map((flag) => {
            const stored = storedFlags.find((f: FeatureFlag) => f.id === flag.id);
            return stored || flag;
          })
        );
      }
    }
  }, [isOpen]);

  const toggleFlag = (id: string) => {
    const updated = flags.map((flag) =>
      flag.id === id ? { ...flag, enabled: !flag.enabled } : flag
    );
    setFlags(updated);
    localStorage.setItem('bul_feature_flags', JSON.stringify(updated));

    // Apply flags
    updated.forEach((flag) => {
      if (flag.id === 'debug-mode') {
        if (flag.enabled) {
          document.documentElement.classList.add('debug-mode');
        } else {
          document.documentElement.classList.remove('debug-mode');
        }
      }
    });
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 300 }}
        animate={{ x: 0 }}
        exit={{ x: 300 }}
        className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 flex flex-col"
      >
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FiFlag size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Feature Flags</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <FiX size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {flags.map((flag) => (
            <div
              key={flag.id}
              className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                    {flag.name}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {flag.description}
                  </p>
                </div>
                <button
                  onClick={() => toggleFlag(flag.id)}
                  className="ml-4"
                >
                  {flag.enabled ? (
                    <FiToggleRight size={24} className="text-primary-600" />
                  ) : (
                    <FiToggleLeft size={24} className="text-gray-400" />
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/20">
          <p className="text-xs text-yellow-800 dark:text-yellow-200">
            ⚠️ Las características experimentales pueden ser inestables
          </p>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


