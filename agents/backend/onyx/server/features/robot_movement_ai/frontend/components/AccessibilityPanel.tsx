'use client';

import { useState } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { Accessibility, Eye, Volume2, MousePointer2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function AccessibilityPanel() {
  const [fontSize, setFontSize] = useLocalStorage('accessibility-font-size', 16);
  const [highContrast, setHighContrast] = useLocalStorage('accessibility-high-contrast', false);
  const [screenReader, setScreenReader] = useLocalStorage('accessibility-screen-reader', false);
  const [reducedMotion, setReducedMotion] = useLocalStorage('accessibility-reduced-motion', false);

  const handleFontSizeChange = (size: number) => {
    setFontSize(size);
    document.documentElement.style.fontSize = `${size}px`;
    toast.success(`Tamaño de fuente: ${size}px`);
  };

  const handleHighContrast = (enabled: boolean) => {
    setHighContrast(enabled);
    if (enabled) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
    toast.success(`Alto contraste: ${enabled ? 'activado' : 'desactivado'}`);
  };

  const handleReducedMotion = (enabled: boolean) => {
    setReducedMotion(enabled);
    if (enabled) {
      document.documentElement.classList.add('reduced-motion');
    } else {
      document.documentElement.classList.remove('reduced-motion');
    }
    toast.success(`Movimiento reducido: ${enabled ? 'activado' : 'desactivado'}`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Accessibility className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Accesibilidad</h3>
        </div>

        {/* Font Size */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Eye className="w-5 h-5 text-blue-400" />
            <h4 className="text-sm font-medium text-gray-300">Tamaño de Fuente</h4>
          </div>
          <div className="flex items-center gap-4">
            <input
              type="range"
              min="12"
              max="24"
              value={fontSize}
              onChange={(e) => handleFontSizeChange(parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="text-white font-mono w-12 text-right">{fontSize}px</span>
          </div>
          <div className="flex gap-2 mt-2">
            {[12, 14, 16, 18, 20, 24].map((size) => (
              <button
                key={size}
                onClick={() => handleFontSizeChange(size)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  fontSize === size
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {size}px
              </button>
            ))}
          </div>
        </div>

        {/* High Contrast */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Eye className="w-5 h-5 text-green-400" />
              <h4 className="text-sm font-medium text-gray-300">Alto Contraste</h4>
            </div>
            <button
              onClick={() => handleHighContrast(!highContrast)}
              className={`relative w-14 h-7 rounded-full transition-colors ${
                highContrast ? 'bg-primary-600' : 'bg-gray-600'
              }`}
            >
              <div
                className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${
                  highContrast ? 'translate-x-7' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Aumenta el contraste de colores para mejor visibilidad
          </p>
        </div>

        {/* Reduced Motion */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MousePointer2 className="w-5 h-5 text-purple-400" />
              <h4 className="text-sm font-medium text-gray-300">Movimiento Reducido</h4>
            </div>
            <button
              onClick={() => handleReducedMotion(!reducedMotion)}
              className={`relative w-14 h-7 rounded-full transition-colors ${
                reducedMotion ? 'bg-primary-600' : 'bg-gray-600'
              }`}
            >
              <div
                className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${
                  reducedMotion ? 'translate-x-7' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Reduce animaciones y transiciones
          </p>
        </div>

        {/* Screen Reader */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Volume2 className="w-5 h-5 text-yellow-400" />
              <h4 className="text-sm font-medium text-gray-300">Lector de Pantalla</h4>
            </div>
            <button
              onClick={() => {
                setScreenReader(!screenReader);
                toast.info('Lector de pantalla: ' + (!screenReader ? 'activado' : 'desactivado'));
              }}
              className={`relative w-14 h-7 rounded-full transition-colors ${
                screenReader ? 'bg-primary-600' : 'bg-gray-600'
              }`}
            >
              <div
                className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${
                  screenReader ? 'translate-x-7' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Optimiza la interfaz para lectores de pantalla
          </p>
        </div>

        {/* Keyboard Shortcuts Info */}
        <div className="p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">Atajos de Teclado</h4>
          <ul className="text-xs text-gray-300 space-y-1">
            <li>• <kbd className="px-1 py-0.5 bg-gray-700 rounded">Tab</kbd> - Navegar entre elementos</li>
            <li>• <kbd className="px-1 py-0.5 bg-gray-700 rounded">Enter</kbd> - Activar elemento</li>
            <li>• <kbd className="px-1 py-0.5 bg-gray-700 rounded">Esc</kbd> - Cerrar modales</li>
            <li>• <kbd className="px-1 py-0.5 bg-gray-700 rounded">Ctrl+K</kbd> - Búsqueda global</li>
          </ul>
        </div>
      </div>
    </div>
  );
}


