'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { FileText, Loader2, CheckCircle, XCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { type Track } from '@/lib/api/music-api';

interface MusicBatchOperationsProps {
  tracks: Track[];
}

export function MusicBatchOperations({ tracks }: MusicBatchOperationsProps) {
  const [selectedTracks, setSelectedTracks] = useState<string[]>([]);
  const [operation, setOperation] = useState<'analyze' | 'favorite' | 'export'>('analyze');

  const batchMutation = useMutation({
    mutationFn: async (params: { operation: string; trackIds: string[] }) => {
      switch (params.operation) {
        case 'analyze':
          return await Promise.all(
            params.trackIds.map((id) => musicApiService.analyzeTrack(id, undefined, false))
          );
        case 'favorite':
          return await Promise.all(
            params.trackIds.map((id) =>
              musicApiService.addToFavorites('user123', id, 'Track', ['Artist'])
            )
          );
        default:
          return [];
      }
    },
    onSuccess: (data) => {
      toast.success(`Operación completada en ${data.length} canciones`);
      setSelectedTracks([]);
    },
    onError: () => {
      toast.error('Error en operación en lote');
    },
  });

  const handleSelectAll = () => {
    if (selectedTracks.length === tracks.length) {
      setSelectedTracks([]);
    } else {
      setSelectedTracks(tracks.map((t) => t.id));
    }
  };

  const handleToggleTrack = (trackId: string) => {
    setSelectedTracks((prev) =>
      prev.includes(trackId) ? prev.filter((id) => id !== trackId) : [...prev, trackId]
    );
  };

  const handleExecute = () => {
    if (selectedTracks.length === 0) {
      toast.error('Selecciona al menos una canción');
      return;
    }
    batchMutation.mutate({ operation, trackIds: selectedTracks });
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Operaciones en Lote</h3>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">
            {selectedTracks.length} de {tracks.length} seleccionadas
          </span>
          <button
            onClick={handleSelectAll}
            className="px-3 py-1 text-sm bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
          >
            {selectedTracks.length === tracks.length ? 'Deseleccionar todas' : 'Seleccionar todas'}
          </button>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">Operación</label>
          <select
            value={operation}
            onChange={(e) => setOperation(e.target.value as any)}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
          >
            <option value="analyze">Analizar</option>
            <option value="favorite">Agregar a Favoritos</option>
            <option value="export">Exportar</option>
          </select>
        </div>

        <div className="max-h-64 overflow-y-auto space-y-2">
          {tracks.map((track) => {
            const isSelected = selectedTracks.includes(track.id);
            return (
              <button
                key={track.id}
                onClick={() => handleToggleTrack(track.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors text-left ${
                  isSelected
                    ? 'bg-purple-600/30 border border-purple-500'
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                  isSelected ? 'bg-purple-600 border-purple-500' : 'border-gray-400'
                }`}>
                  {isSelected && <CheckCircle className="w-4 h-4 text-white" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">{track.name}</p>
                  <p className="text-sm text-gray-300 truncate">
                    {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        <button
          onClick={handleExecute}
          disabled={batchMutation.isPending || selectedTracks.length === 0}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {batchMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Procesando...
            </>
          ) : (
            <>
              <FileText className="w-5 h-5" />
              Ejecutar en {selectedTracks.length} canción{selectedTracks.length !== 1 ? 'es' : ''}
            </>
          )}
        </button>
      </div>
    </div>
  );
}


