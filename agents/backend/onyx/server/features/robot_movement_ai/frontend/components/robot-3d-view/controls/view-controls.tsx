/**
 * View Controls Component
 * @module robot-3d-view/controls/view-controls
 */

'use client';

import { memo, useEffect } from 'react';
import type { SceneConfig, CameraPreset } from '../types';
import { ariaLabels, keyboardShortcuts, getKeyboardShortcut } from '../utils/accessibility';

/**
 * Props for ViewControls component
 */
interface ViewControlsProps {
  config: SceneConfig;
  onToggleStats: () => void;
  onToggleGizmo: () => void;
  onToggleStars: () => void;
  onToggleWaypoints: () => void;
  onToggleGrid: () => void;
  onToggleObjects: () => void;
  onToggleAutoRotate: () => void;
  onSetCameraPreset: (preset: CameraPreset) => void;
}

/**
 * View Controls Component
 * 
 * Provides UI controls for toggling various view options and camera presets.
 * 
 * @param props - Control handlers and configuration
 * @returns View controls component
 */
export const ViewControls = memo(({
  config,
  onToggleStats,
  onToggleGizmo,
  onToggleStars,
  onToggleWaypoints,
  onToggleGrid,
  onToggleObjects,
  onToggleAutoRotate,
  onSetCameraPreset,
}: ViewControlsProps) => {
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Ignore if typing in input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (event.code) {
        case keyboardShortcuts.toggleStats:
          event.preventDefault();
          onToggleStats();
          break;
        case keyboardShortcuts.toggleGizmo:
          event.preventDefault();
          onToggleGizmo();
          break;
        case keyboardShortcuts.toggleGrid:
          event.preventDefault();
          onToggleGrid();
          break;
        case keyboardShortcuts.toggleObjects:
          event.preventDefault();
          onToggleObjects();
          break;
        case keyboardShortcuts.toggleAutoRotate:
          event.preventDefault();
          onToggleAutoRotate();
          break;
        case keyboardShortcuts.cameraFront:
          event.preventDefault();
          onSetCameraPreset('front');
          break;
        case keyboardShortcuts.cameraTop:
          event.preventDefault();
          onSetCameraPreset('top');
          break;
        case keyboardShortcuts.cameraSide:
          event.preventDefault();
          onSetCameraPreset('side');
          break;
        case keyboardShortcuts.cameraIso:
          event.preventDefault();
          onSetCameraPreset('iso');
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [
    onToggleStats,
    onToggleGizmo,
    onToggleGrid,
    onToggleObjects,
    onToggleAutoRotate,
    onSetCameraPreset,
  ]);

  const buttonClassName =
    'px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900';

  return (
    <div
      className="absolute top-4 right-4 flex flex-col gap-2 max-h-[80vh] overflow-y-auto"
      role="toolbar"
      aria-label={ariaLabels.controls}
    >
      <button
        onClick={onToggleStats}
        className={buttonClassName}
        title={`${config.showStats ? 'Ocultar' : 'Mostrar'} estadísticas de rendimiento (${getKeyboardShortcut('toggleStats')})`}
        aria-label={ariaLabels.toggleStats}
      >
        {config.showStats ? 'Ocultar' : 'Mostrar'} Stats
      </button>
      <button onClick={onToggleAutoRotate} className={buttonClassName} title="Rotación automática">
        {config.autoRotate ? '⏸' : '▶'} Auto
      </button>
      <button onClick={onToggleGizmo} className={buttonClassName} title="Mostrar gizmo de navegación">
        {config.showGizmo ? '🔲' : '⬜'} Gizmo
      </button>
      <button onClick={onToggleStars} className={buttonClassName} title="Mostrar fondo estrellado">
        {config.showStars ? '⭐' : '🌌'} Sky
      </button>
      <button onClick={onToggleWaypoints} className={buttonClassName} title="Mostrar puntos de ruta">
        {config.showWaypoints ? '📍' : '🔲'} Waypoints
      </button>
      <button onClick={onToggleGrid} className={buttonClassName} title="Mostrar/Ocultar grid">
        {config.showGrid ? '⊞' : '⊠'} Grid
      </button>
      <button onClick={onToggleObjects} className={buttonClassName} title="Mostrar/Ocultar objetos de entorno">
        {config.showObjects ? '📦' : '📭'} Objetos
      </button>

      {/* Camera Presets */}
      <div className="pt-2 border-t border-gray-700">
        <div className="text-[10px] text-gray-400 mb-1 px-1">Cámara:</div>
        <div className="grid grid-cols-2 gap-1">
          {(['front', 'top', 'side', 'iso'] as const).map((preset) => (
            <button
              key={preset}
              onClick={() => onSetCameraPreset(preset)}
              className={`px-2 py-1 text-[10px] rounded ${
                config.cameraPreset === preset ? 'bg-blue-600' : 'bg-gray-700/50'
              } hover:bg-gray-600 transition-all`}
              title={`Vista ${preset === 'iso' ? 'isométrica' : preset === 'front' ? 'frontal' : preset === 'top' ? 'superior' : 'lateral'}`}
            >
              {preset === 'iso' ? 'ISO' : preset.charAt(0).toUpperCase() + preset.slice(1)}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
});

ViewControls.displayName = 'ViewControls';

