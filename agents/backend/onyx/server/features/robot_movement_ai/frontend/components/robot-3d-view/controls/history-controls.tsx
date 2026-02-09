/**
 * History Controls Component
 * @module robot-3d-view/controls/history-controls
 */

'use client';

import { memo } from 'react';
import { useHistory } from '../hooks/use-history';
import { notify } from '../utils/notifications';

/**
 * Props for HistoryControls component
 */
interface HistoryControlsProps {
  onConfigChange: (config: ReturnType<typeof useHistory>['undo'] extends () => infer R ? R : never) => void;
}

/**
 * History Controls Component
 * 
 * Provides undo/redo functionality for configuration changes.
 * 
 * @param props - Component props
 * @returns History controls component
 */
export const HistoryControls = memo(({ onConfigChange }: HistoryControlsProps) => {
  const { canUndo, canRedo, undo, redo } = useHistory();

  const handleUndo = () => {
    const config = undo();
    if (config) {
      onConfigChange(config);
      notify.info('Cambio deshecho');
    }
  };

  const handleRedo = () => {
    const config = redo();
    if (config) {
      onConfigChange(config);
      notify.info('Cambio rehecho');
    }
  };

  return (
    <div className="absolute top-20 right-4 z-40 flex gap-2">
      <button
        onClick={handleUndo}
        disabled={!canUndo}
        className={`
          px-3 py-2 bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg
          text-white text-xs font-medium transition-all shadow-lg
          ${canUndo ? 'hover:bg-gray-700/95 cursor-pointer' : 'opacity-50 cursor-not-allowed'}
        `}
        title="Deshacer (Ctrl+Z)"
        aria-label="Deshacer último cambio"
      >
        ↶ Deshacer
      </button>
      <button
        onClick={handleRedo}
        disabled={!canRedo}
        className={`
          px-3 py-2 bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg
          text-white text-xs font-medium transition-all shadow-lg
          ${canRedo ? 'hover:bg-gray-700/95 cursor-pointer' : 'opacity-50 cursor-not-allowed'}
        `}
        title="Rehacer (Ctrl+Shift+Z)"
        aria-label="Rehacer último cambio deshecho"
      >
        ↷ Rehacer
      </button>
    </div>
  );
});

HistoryControls.displayName = 'HistoryControls';



