'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiEye, FiType, FiMousePointer } from 'react-icons/fi';

interface AccessibilityToolsProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AccessibilityTools({ isOpen, onClose }: AccessibilityToolsProps) {
  const [fontSize, setFontSize] = useState(16);
  const [highContrast, setHighContrast] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [focusVisible, setFocusVisible] = useState(true);

  useEffect(() => {
    if (isOpen) {
      const stored = {
        fontSize: localStorage.getItem('bul_a11y_fontSize') || '16',
        highContrast: localStorage.getItem('bul_a11y_highContrast') === 'true',
        reducedMotion: localStorage.getItem('bul_a11y_reducedMotion') === 'true',
        focusVisible: localStorage.getItem('bul_a11y_focusVisible') !== 'false',
      };
      setFontSize(parseInt(stored.fontSize));
      setHighContrast(stored.highContrast);
      setReducedMotion(stored.reducedMotion);
      setFocusVisible(stored.focusVisible);
    }
  }, [isOpen]);

  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}px`;
    localStorage.setItem('bul_a11y_fontSize', fontSize.toString());
  }, [fontSize]);

  useEffect(() => {
    if (highContrast) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
    localStorage.setItem('bul_a11y_highContrast', highContrast.toString());
  }, [highContrast]);

  useEffect(() => {
    if (reducedMotion) {
      document.documentElement.classList.add('reduce-motion');
    } else {
      document.documentElement.classList.remove('reduce-motion');
    }
    localStorage.setItem('bul_a11y_reducedMotion', reducedMotion.toString());
  }, [reducedMotion]);

  useEffect(() => {
    if (focusVisible) {
      document.documentElement.classList.add('focus-visible');
    } else {
      document.documentElement.classList.remove('focus-visible');
    }
    localStorage.setItem('bul_a11y_focusVisible', focusVisible.toString());
  }, [focusVisible]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 300 }}
        animate={{ x: 0 }}
        exit={{ x: 300 }}
        className="fixed right-0 top-0 h-full w-80 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 flex flex-col"
      >
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FiEye size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Accesibilidad</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18" />
              <path d="M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <div>
            <label className="flex items-center gap-2 mb-2">
              <FiType size={18} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Tamaño de Fuente Global
              </span>
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="12"
                max="24"
                value={fontSize}
                onChange={(e) => setFontSize(parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm font-medium w-12 text-center">{fontSize}px</span>
            </div>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiEye size={18} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Alto Contraste
                </span>
              </div>
              <input
                type="checkbox"
                checked={highContrast}
                onChange={(e) => setHighContrast(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Mejora el contraste de colores para mejor legibilidad
            </p>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiMousePointer size={18} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Reducir Animaciones
                </span>
              </div>
              <input
                type="checkbox"
                checked={reducedMotion}
                onChange={(e) => setReducedMotion(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Reduce animaciones para usuarios sensibles al movimiento
            </p>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiMousePointer size={18} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Indicador de Foco Visible
                </span>
              </div>
              <input
                type="checkbox"
                checked={focusVisible}
                onChange={(e) => setFocusVisible(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Muestra claramente el elemento enfocado
            </p>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


