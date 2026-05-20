'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiHelpCircle, FiX } from 'react-icons/fi';

interface HelpTip {
  id: string;
  selector: string;
  title: string;
  content: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

const helpTips: HelpTip[] = [
  {
    id: 'generate-query',
    selector: '[data-help="generate-query"]',
    title: 'Consulta de Negocio',
    content: 'Describe aquí qué documento necesitas. Puedes usar plantillas, voz o escribir manualmente.',
    position: 'bottom',
  },
  {
    id: 'templates',
    selector: '[data-help="templates"]',
    title: 'Plantillas',
    content: 'Selecciona una plantilla predefinida para comenzar rápidamente.',
    position: 'right',
  },
  {
    id: 'filters',
    selector: '[data-help="filters"]',
    title: 'Filtros Avanzados',
    content: 'Usa los filtros para encontrar tareas específicas por estado, fecha o prioridad.',
    position: 'bottom',
  },
];

interface ContextualHelpProps {
  enabled: boolean;
}

export default function ContextualHelp({ enabled }: ContextualHelpProps) {
  const [currentTip, setCurrentTip] = useState<HelpTip | null>(null);
  const [dismissedTips, setDismissedTips] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!enabled) return;

    const checkTips = () => {
      for (const tip of helpTips) {
        if (dismissedTips.has(tip.id)) continue;

        const element = document.querySelector(tip.selector);
        if (element && isElementVisible(element)) {
          setCurrentTip(tip);
          return;
        }
      }
    };

    const isElementVisible = (element: Element) => {
      const rect = element.getBoundingClientRect();
      return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= window.innerHeight &&
        rect.right <= window.innerWidth
      );
    };

    checkTips();
    const interval = setInterval(checkTips, 1000);

    return () => clearInterval(interval);
  }, [enabled, dismissedTips]);

  const dismissTip = () => {
    if (currentTip) {
      setDismissedTips((prev) => new Set([...prev, currentTip.id]));
      setCurrentTip(null);
    }
  };

  if (!enabled || !currentTip) return null;

  const element = document.querySelector(currentTip.selector);
  if (!element) return null;

  const rect = element.getBoundingClientRect();
  const position = currentTip.position || 'bottom';

  const getPosition = () => {
    switch (position) {
      case 'top':
        return {
          top: rect.top - 10,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, -100%)',
        };
      case 'bottom':
        return {
          top: rect.bottom + 10,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, 0)',
        };
      case 'left':
        return {
          top: rect.top + rect.height / 2,
          left: rect.left - 10,
          transform: 'translate(-100%, -50%)',
        };
      case 'right':
        return {
          top: rect.top + rect.height / 2,
          left: rect.right + 10,
          transform: 'translate(0, -50%)',
        };
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        style={getPosition()}
        className="fixed z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 max-w-xs"
      >
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-2">
            <FiHelpCircle size={18} className="text-primary-600" />
            <h4 className="font-semibold text-gray-900 dark:text-white text-sm">
              {currentTip.title}
            </h4>
          </div>
          <button onClick={dismissTip} className="btn-icon">
            <FiX size={14} />
          </button>
        </div>
        <p className="text-xs text-gray-600 dark:text-gray-400">{currentTip.content}</p>
      </motion.div>
    </AnimatePresence>
  );
}


