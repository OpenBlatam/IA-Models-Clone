'use client';

import { useState } from 'react';
import { Sliders } from 'lucide-react';

export function Equalizer() {
  const [bands, setBands] = useState({
    bass: 0,
    mid: 0,
    treble: 0,
  });

  const handleBandChange = (band: keyof typeof bands, value: number) => {
    setBands((prev) => ({ ...prev, [band]: value }));
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Sliders className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Ecualizador</h3>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Bajos</span>
            <span className="text-sm text-white">{bands.bass > 0 ? '+' : ''}{bands.bass}dB</span>
          </div>
          <input
            type="range"
            min="-12"
            max="12"
            value={bands.bass}
            onChange={(e) => handleBandChange('bass', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Medios</span>
            <span className="text-sm text-white">{bands.mid > 0 ? '+' : ''}{bands.mid}dB</span>
          </div>
          <input
            type="range"
            min="-12"
            max="12"
            value={bands.mid}
            onChange={(e) => handleBandChange('mid', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Agudos</span>
            <span className="text-sm text-white">{bands.treble > 0 ? '+' : ''}{bands.treble}dB</span>
          </div>
          <input
            type="range"
            min="-12"
            max="12"
            value={bands.treble}
            onChange={(e) => handleBandChange('treble', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>

        <button
          onClick={() => setBands({ bass: 0, mid: 0, treble: 0 })}
          className="w-full px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors text-sm"
        >
          Resetear
        </button>
      </div>
    </div>
  );
}


