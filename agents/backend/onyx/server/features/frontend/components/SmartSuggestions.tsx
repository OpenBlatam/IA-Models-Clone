'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiLightbulb, FiX } from 'react-icons/fi';
import { useAppStore } from '@/store/app-store';

interface Suggestion {
  id: string;
  type: 'tip' | 'feature' | 'optimization';
  title: string;
  message: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export default function SmartSuggestions() {
  const { setActiveView } = useAppStore();
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  useEffect(() => {
    const generateSuggestions = () => {
      const newSuggestions: Suggestion[] = [];

      // Check if user hasn't used templates
      const templatesUsed = localStorage.getItem('bul_templates_used');
      if (!templatesUsed) {
        newSuggestions.push({
          id: 'use-templates',
          type: 'tip',
          title: 'Usa Plantillas',
          message: 'Las plantillas pueden acelerar tu trabajo. Prueba el botón "Usar Plantilla"',
          action: {
            label: 'Ver Plantillas',
            onClick: () => {
              setActiveView('generate');
              localStorage.setItem('bul_templates_used', 'true');
            },
          },
        });
      }

      // Check if user hasn't used favorites
      const favorites = localStorage.getItem('bul_favorites');
      if (!favorites || JSON.parse(favorites).length === 0) {
        newSuggestions.push({
          id: 'use-favorites',
          type: 'tip',
          title: 'Guarda Favoritos',
          message: 'Marca documentos como favoritos para acceso rápido',
        });
      }

      // Check if user hasn't used shortcuts
      const shortcutsUsed = localStorage.getItem('bul_shortcuts_used');
      if (!shortcutsUsed) {
        newSuggestions.push({
          id: 'use-shortcuts',
          type: 'tip',
          title: 'Atajos de Teclado',
          message: 'Presiona Ctrl+/ para ver todos los atajos disponibles',
          action: {
            label: 'Ver Atajos',
            onClick: () => {
              window.dispatchEvent(new CustomEvent('bul_open_shortcuts'));
              localStorage.setItem('bul_shortcuts_used', 'true');
            },
          },
        });
      }

      setSuggestions(newSuggestions.filter((s) => !dismissed.has(s.id)));
    };

    generateSuggestions();
    const interval = setInterval(generateSuggestions, 60000); // Check every minute
    return () => clearInterval(interval);
  }, [dismissed]);

  const dismissSuggestion = (id: string) => {
    setDismissed((prev) => {
      const updated = new Set(prev);
      updated.add(id);
      return updated;
    });
    setSuggestions((prev) => prev.filter((s) => s.id !== id));
  };

  if (suggestions.length === 0) return null;

  return (
    <AnimatePresence>
      {suggestions.slice(0, 1).map((suggestion) => (
        <motion.div
          key={suggestion.id}
          initial={{ x: 400, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 400, opacity: 0 }}
          className="fixed bottom-4 right-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg shadow-lg p-4 max-w-sm z-50"
        >
          <div className="flex items-start gap-3">
            <FiLightbulb size={24} className="text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-yellow-900 dark:text-yellow-200 mb-1">
                {suggestion.title}
              </h4>
              <p className="text-sm text-yellow-800 dark:text-yellow-300 mb-3">
                {suggestion.message}
              </p>
              {suggestion.action && (
                <button
                  onClick={suggestion.action.onClick}
                  className="text-sm text-yellow-700 dark:text-yellow-400 hover:underline font-medium"
                >
                  {suggestion.action.label} →
                </button>
              )}
            </div>
            <button
              onClick={() => dismissSuggestion(suggestion.id)}
              className="btn-icon text-yellow-600 dark:text-yellow-400"
            >
              <FiX size={18} />
            </button>
          </div>
        </motion.div>
      ))}
    </AnimatePresence>
  );
}

