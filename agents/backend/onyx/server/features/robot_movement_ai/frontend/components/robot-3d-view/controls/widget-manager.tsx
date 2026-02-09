/**
 * Widget Manager Component
 * @module robot-3d-view/controls/widget-manager
 */

'use client';

import { memo, useState } from 'react';
import { useWidgets } from '../hooks/use-widgets';
import { notify } from '../utils/notifications';
import { Tooltip } from '../components/tooltip';
import { createTooltip } from '../utils/tooltips';

/**
 * Widget Manager Component
 * 
 * Provides UI for managing customizable widgets.
 * 
 * @returns Widget manager component
 */
export const WidgetManager = memo(() => {
  const { widgets, toggleWidget, updateWidget, resetToDefaults } = useWidgets();
  const [isOpen, setIsOpen] = useState(false);

  if (!isOpen) {
    return (
      <Tooltip tooltip={createTooltip('Gestionar widgets')}>
        <button
          onClick={() => setIsOpen(true)}
          className="absolute bottom-4 right-4 z-50 px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg"
          title="Gestionar widgets"
          aria-label="Abrir gestor de widgets"
        >
          🎛️ Widgets
        </button>
      </Tooltip>
    );
  }

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={() => setIsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Gestor de widgets"
    >
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Gestor de Widgets</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Cerrar gestor de widgets"
          >
            ✕
          </button>
        </div>

        <div className="space-y-2">
          {widgets.map((widget) => (
            <div
              key={widget.id}
              className="flex items-center justify-between p-3 bg-gray-700/50 rounded"
            >
              <div className="flex-1">
                <div className="text-sm font-medium text-white">{widget.title}</div>
                <div className="text-xs text-gray-400">{widget.type}</div>
              </div>
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={widget.visible}
                    onChange={() => {
                      toggleWidget(widget.id);
                      notify.info(`Widget ${widget.visible ? 'ocultado' : 'mostrado'}`);
                    }}
                    className="w-4 h-4"
                  />
                  <span className="text-xs text-gray-300">Visible</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={widget.pinned}
                    onChange={() => {
                      updateWidget(widget.id, { pinned: !widget.pinned });
                      notify.info(`Widget ${widget.pinned ? 'desanclado' : 'anclado'}`);
                    }}
                    className="w-4 h-4"
                  />
                  <span className="text-xs text-gray-300">Anclado</span>
                </label>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 flex gap-2">
          <button
            onClick={() => {
              resetToDefaults();
              notify.success('Widgets restaurados a valores por defecto');
            }}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-white text-sm transition-colors"
          >
            Restaurar por defecto
          </button>
        </div>
      </div>
    </div>
  );
});

WidgetManager.displayName = 'WidgetManager';



