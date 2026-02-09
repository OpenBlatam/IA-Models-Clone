'use client';

import { useState } from 'react';
import { CheckSquare, Square, Trash2, Download, Share2 } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface BulkActionsProps {
  tracks: Track[];
  selectedTracks: string[];
  onSelectionChange: (trackIds: string[]) => void;
  onBulkAction: (action: string, trackIds: string[]) => void;
}

export function BulkActions({
  tracks,
  selectedTracks,
  onSelectionChange,
  onBulkAction,
}: BulkActionsProps) {
  const [isSelectMode, setIsSelectMode] = useState(false);

  const toggleSelect = (trackId: string) => {
    if (selectedTracks.includes(trackId)) {
      onSelectionChange(selectedTracks.filter((id) => id !== trackId));
    } else {
      onSelectionChange([...selectedTracks, trackId]);
    }
  };

  const selectAll = () => {
    if (selectedTracks.length === tracks.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange(tracks.map((t) => t.id));
    }
  };

  if (!isSelectMode) {
    return (
      <button
        onClick={() => setIsSelectMode(true)}
        className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
      >
        <CheckSquare className="w-5 h-5" />
        <span>Seleccionar Múltiples</span>
      </button>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <button
            onClick={selectAll}
            className="flex items-center gap-2 px-3 py-1 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors text-sm"
          >
            {selectedTracks.length === tracks.length ? (
              <CheckSquare className="w-4 h-4" />
            ) : (
              <Square className="w-4 h-4" />
            )}
            <span>Seleccionar Todo</span>
          </button>
          <span className="text-sm text-gray-300">
            {selectedTracks.length} seleccionados
          </span>
        </div>
        <button
          onClick={() => {
            setIsSelectMode(false);
            onSelectionChange([]);
          }}
          className="text-gray-400 hover:text-white"
        >
          Cancelar
        </button>
      </div>

      {selectedTracks.length > 0 && (
        <div className="flex gap-2">
          <button
            onClick={() => onBulkAction('delete', selectedTracks)}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Eliminar
          </button>
          <button
            onClick={() => onBulkAction('download', selectedTracks)}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
          >
            <Download className="w-4 h-4" />
            Descargar
          </button>
          <button
            onClick={() => onBulkAction('share', selectedTracks)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <Share2 className="w-4 h-4" />
            Compartir
          </button>
        </div>
      )}

      <div className="mt-4 space-y-2 max-h-64 overflow-y-auto">
        {tracks.map((track) => (
          <button
            key={track.id}
            onClick={() => toggleSelect(track.id)}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors ${
              selectedTracks.includes(track.id)
                ? 'bg-purple-500/30 border-2 border-purple-400'
                : 'bg-white/5 hover:bg-white/10'
            }`}
          >
            {selectedTracks.includes(track.id) ? (
              <CheckSquare className="w-5 h-5 text-purple-300" />
            ) : (
              <Square className="w-5 h-5 text-gray-400" />
            )}
            <div className="flex-1 text-left">
              <p className="text-white font-medium">{track.name}</p>
              <p className="text-sm text-gray-300">{track.artists.join(', ')}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}


