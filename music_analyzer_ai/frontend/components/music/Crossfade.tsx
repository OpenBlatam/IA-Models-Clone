'use client';

import { useState } from 'react';
import { Volume2 } from 'lucide-react';

export function Crossfade() {
  const [crossfadeDuration, setCrossfadeDuration] = useState(3);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Volume2 className="w-4 h-4 text-purple-300" />
        <h3 className="text-sm font-semibold text-white">Crossfade</h3>
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Duración</span>
            <span className="text-sm text-white">{crossfadeDuration}s</span>
          </div>
          <input
            type="range"
            min="0"
            max="12"
            step="1"
            value={crossfadeDuration}
            onChange={(e) => setCrossfadeDuration(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>0s</span>
            <span>12s</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="crossfade-enabled"
            defaultChecked
            className="w-4 h-4 rounded accent-purple-500"
          />
          <label htmlFor="crossfade-enabled" className="text-sm text-gray-300">
            Activar crossfade
          </label>
        </div>
      </div>
    </div>
  );
}


