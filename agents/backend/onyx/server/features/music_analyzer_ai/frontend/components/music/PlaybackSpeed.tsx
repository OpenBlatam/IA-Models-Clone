'use client';

import { useState } from 'react';
import { Gauge } from 'lucide-react';

interface PlaybackSpeedProps {
  onSpeedChange: (speed: number) => void;
}

export function PlaybackSpeed({ onSpeedChange }: PlaybackSpeedProps) {
  const [speed, setSpeed] = useState(1);

  const speeds = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    onSpeedChange(newSpeed);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Gauge className="w-4 h-4 text-purple-300" />
        <h3 className="text-sm font-semibold text-white">Velocidad</h3>
      </div>

      <div className="flex flex-wrap gap-2">
        {speeds.map((s) => (
          <button
            key={s}
            onClick={() => handleSpeedChange(s)}
            className={`px-3 py-1 rounded-lg text-sm transition-colors ${
              speed === s
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      <div className="mt-3">
        <input
          type="range"
          min="0.5"
          max="2"
          step="0.1"
          value={speed}
          onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>0.5x</span>
          <span className="text-white font-medium">{speed.toFixed(1)}x</span>
          <span>2x</span>
        </div>
      </div>
    </div>
  );
}


