'use client';

import { useEffect, useState } from 'react';
import { useRecordingStore } from '@/lib/store/recordingStore';
import { useRobotStore } from '@/lib/store/robotStore';
import { CircleDot, Square, Play, Trash2, Download, Upload, Save } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function RecordingPanel() {
  const {
    isRecording,
    isPlaying,
    records,
    startRecording,
    stopRecording,
    clearRecords,
    startPlayback,
    stopPlayback,
    nextRecord,
  } = useRecordingStore();
  const { moveTo, stop, currentPosition } = useRobotStore();
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  useEffect(() => {
    if (isPlaying && records.length > 0) {
      const interval = setInterval(() => {
        const record = nextRecord();
        if (record) {
          if (record.action === 'move') {
            moveTo(record.position);
          } else if (record.action === 'stop') {
            stop();
          }
        } else {
          stopPlayback();
          toast.success('Reproducción completada');
        }
      }, 1000 / playbackSpeed);

      return () => clearInterval(interval);
    }
  }, [isPlaying, records, playbackSpeed, nextRecord, moveTo, stop, stopPlayback]);

  const handleRecord = () => {
    if (isRecording) {
      stopRecording();
      toast.success(`Grabación detenida. ${records.length} movimientos guardados.`);
    } else {
      startRecording();
      toast.info('Grabación iniciada');
    }
  };

  const handlePlay = () => {
    if (records.length === 0) {
      toast.error('No hay movimientos grabados');
      return;
    }
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
      toast.info('Reproducción iniciada');
    }
  };

  const handleSave = () => {
    if (records.length === 0) {
      toast.error('No hay movimientos para guardar');
      return;
    }
    const dataStr = JSON.stringify(records, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `robot-recording-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Grabación guardada');
  };

  const handleLoad = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const data = JSON.parse(event.target?.result as string);
            useRecordingStore.setState({ records: data });
            toast.success('Grabación cargada');
          } catch (error) {
            toast.error('Error al cargar el archivo');
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
      <div className="flex items-center gap-2 mb-6">
        <CircleDot className={`w-5 h-5 text-tesla-blue ${isRecording ? 'animate-pulse' : ''}`} />
        <h3 className="text-lg font-semibold text-tesla-black">Grabación y Reproducción</h3>
      </div>

      <div className="space-y-6">
        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={handleRecord}
            className={`flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-md font-medium transition-all min-h-[44px] ${
              isRecording
                ? 'bg-red-600 hover:bg-red-700 text-white shadow-md'
                : 'bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black'
            }`}
            aria-label={isRecording ? 'Detener grabación' : 'Iniciar grabación'}
          >
            {isRecording ? (
              <>
                <Square className="w-4 h-4" />
                Detener Grabación
              </>
            ) : (
              <>
                <CircleDot className="w-4 h-4" />
                Iniciar Grabación
              </>
            )}
          </button>
          <button
            onClick={handlePlay}
            disabled={records.length === 0}
            className={`flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-md font-medium transition-all min-h-[44px] ${
              isPlaying
                ? 'bg-yellow-600 hover:bg-yellow-700 text-white shadow-md'
                : 'bg-green-600 hover:bg-green-700 text-white'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
            aria-label={isPlaying ? 'Detener reproducción' : 'Reproducir grabación'}
          >
            {isPlaying ? (
              <>
                <Square className="w-4 h-4" />
                Detener
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Reproducir
              </>
            )}
          </button>
        </div>

        {/* Playback Speed */}
        {isPlaying && (
          <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
            <label className="block text-sm font-medium text-tesla-black mb-3">
              Velocidad de Reproducción: <span className="text-tesla-blue font-semibold">{playbackSpeed}x</span>
            </label>
            <input
              type="range"
              min="0.5"
              max="3"
              step="0.1"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-tesla-blue"
              aria-label="Velocidad de reproducción"
            />
            <div className="flex justify-between text-xs text-tesla-gray-dark mt-1">
              <span>0.5x</span>
              <span>3x</span>
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
            <p className="text-sm text-tesla-gray-dark mb-1 font-medium">Movimientos Grabados</p>
            <p className="text-2xl font-bold text-tesla-black">{records.length}</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
            <p className="text-sm text-tesla-gray-dark mb-1 font-medium">Estado</p>
            <p className={`text-lg font-semibold ${
              isRecording ? 'text-red-600' : isPlaying ? 'text-yellow-600' : 'text-tesla-gray-dark'
            }`}>
              {isRecording ? 'Grabando...' : isPlaying ? 'Reproduciendo...' : 'Inactivo'}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleSave}
            disabled={records.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-tesla-blue hover:bg-opacity-90 text-white rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium min-h-[44px]"
            aria-label="Guardar grabación"
          >
            <Download className="w-4 h-4" />
            Guardar
          </button>
          <button
            onClick={handleLoad}
            className="flex items-center gap-2 px-4 py-2 bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium min-h-[44px]"
            aria-label="Cargar grabación"
          >
            <Upload className="w-4 h-4" />
            Cargar
          </button>
          <button
            onClick={clearRecords}
            disabled={records.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium min-h-[44px]"
            aria-label="Limpiar grabaciones"
          >
            <Trash2 className="w-4 h-4" />
            Limpiar
          </button>
        </div>

        {/* Records List */}
        {records.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-semibold text-tesla-black mb-3">Movimientos Grabados</h4>
            <div className="max-h-48 overflow-y-auto space-y-2 scrollbar-hide">
              {records.map((record, index) => (
                <div
                  key={index}
                  className="p-3 bg-gray-50 rounded-md border border-gray-200 text-sm text-tesla-black hover:shadow-sm transition-all"
                >
                  <span className="font-mono text-xs">
                    {record.action} - ({record.position.x.toFixed(2)}, {record.position.y.toFixed(2)}, {record.position.z.toFixed(2)})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

