/**
 * Help Overlay Component
 * @module robot-3d-view/components/help-overlay
 */

'use client';

import { memo, useState } from 'react';
import { SHORTCUTS, getShortcutDisplay } from '../utils/shortcuts';

/**
 * Help Overlay Component
 * 
 * Displays keyboard shortcuts and help information.
 * 
 * @returns Help overlay component
 */
export const HelpOverlay = memo(() => {
  const [isOpen, setIsOpen] = useState(false);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="absolute bottom-4 right-20 z-50 px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg"
        title="Mostrar ayuda"
        aria-label="Mostrar ayuda y atajos de teclado"
      >
        ❓ Ayuda
      </button>
    );
  }

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={() => setIsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Ayuda y atajos de teclado"
    >
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Ayuda y Atajos de Teclado</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Cerrar ayuda"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2">Atajos de Teclado</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {SHORTCUTS.map((shortcut) => (
                <div
                  key={shortcut.action}
                  className="flex items-center justify-between p-2 bg-gray-700/50 rounded"
                >
                  <span className="text-sm text-gray-300">{shortcut.description}</span>
                  <kbd className="px-2 py-1 bg-gray-600 rounded text-xs font-mono text-white">
                    {shortcut.key.toUpperCase()}
                  </kbd>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">Controles del Mouse</h3>
            <ul className="space-y-1 text-sm text-gray-300">
              <li>🖱️ <strong>Arrastrar:</strong> Rotar la cámara</li>
              <li>🔍 <strong>Rueda:</strong> Zoom in/out</li>
              <li>👆 <strong>Click derecho + arrastrar:</strong> Pan (mover)</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">Información</h3>
            <p className="text-sm text-gray-300">
              Usa los controles en la parte superior derecha para personalizar la vista.
              Puedes exportar e importar configuraciones, cambiar temas y aplicar presets.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
});

HelpOverlay.displayName = 'HelpOverlay';



