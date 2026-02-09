/**
 * Recording Controls Component
 * @module robot-3d-view/controls/recording-controls
 */

'use client';

import { memo, useRef } from 'react';
import { useRecording } from '../hooks/use-recording';
import { notify } from '../utils/notifications';

/**
 * Props for RecordingControls component
 */
interface RecordingControlsProps {
  onRecordingChange?: (isRecording: boolean) => void;
}

/**
 * Recording Controls Component
 * 
 * Provides controls for recording and playing back robot movements.
 * 
 * @param props - Component props
 * @returns Recording controls component
 */
export const RecordingControls = memo(({ onRecordingChange }: RecordingControlsProps) => {
  const {
    isRecording,
    recordings,
    startRecording,
    stopRecording,
    deleteRecording,
    exportRecording,
  } = useRecording();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleStart = () => {
    startRecording();
    notify.success('Grabación iniciada');
    onRecordingChange?.(true);
  };

  const handleStop = () => {
    const recording = stopRecording();
    if (recording) {
      notify.success(`Grabación completada: ${recording.frames.length} frames`);
    }
    onRecordingChange?.(false);
  };

  const handleExport = (id: string) => {
    exportRecording(id);
    notify.success('Grabación exportada');
  };

  return (
    <div className="absolute top-32 right-4 z-40">
      <div className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-2 shadow-lg">
        <div className="text-[10px] text-gray-400 mb-2 px-2">Grabación:</div>
        <div className="flex flex-col gap-1">
          {!isRecording ? (
            <button
              onClick={handleStart}
              className="px-2 py-1 text-[10px] rounded bg-red-600/80 hover:bg-red-600 transition-all text-white"
              title="Iniciar grabación"
            >
              🔴 Grabar
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 transition-all text-white"
              title="Detener grabación"
            >
              ⏹ Detener
            </button>
          )}

          {recordings.length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-700">
              <div className="text-[10px] text-gray-400 mb-1 px-2">
                Grabaciones ({recordings.length}):
              </div>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {recordings.map((recording) => (
                  <div
                    key={recording.id}
                    className="flex items-center gap-1 px-2 py-1 bg-gray-700/30 rounded text-[10px]"
                  >
                    <span className="flex-1 truncate text-gray-300">
                      {recording.name}
                    </span>
                    <button
                      onClick={() => handleExport(recording.id)}
                      className="text-blue-400 hover:text-blue-300"
                      title="Exportar"
                    >
                      📥
                    </button>
                    <button
                      onClick={() => {
                        deleteRecording(recording.id);
                        notify.info('Grabación eliminada');
                      }}
                      className="text-red-400 hover:text-red-300"
                      title="Eliminar"
                    >
                      🗑
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

RecordingControls.displayName = 'RecordingControls';



