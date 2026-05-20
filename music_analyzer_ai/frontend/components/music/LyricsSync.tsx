'use client';

import { useState, useEffect } from 'react';
import { Music, Clock } from 'lucide-react';

interface LyricsSyncProps {
  lyrics: string;
  currentTime: number;
  duration: number;
}

export function LyricsSync({ lyrics, currentTime, duration }: LyricsSyncProps) {
  const [syncedLyrics, setSyncedLyrics] = useState<Array<{ time: number; text: string }>>([]);
  const [currentLine, setCurrentLine] = useState(0);

  useEffect(() => {
    // Parsear letras sincronizadas (formato LRC)
    const lines = lyrics.split('\n');
    const parsed: Array<{ time: number; text: string }> = [];

    lines.forEach((line) => {
      const match = line.match(/\[(\d{2}):(\d{2})\.(\d{2})\](.*)/);
      if (match) {
        const minutes = parseInt(match[1]);
        const seconds = parseInt(match[2]);
        const centiseconds = parseInt(match[3]);
        const time = minutes * 60 + seconds + centiseconds / 100;
        parsed.push({ time, text: match[4].trim() });
      }
    });

    setSyncedLyrics(parsed);
  }, [lyrics]);

  useEffect(() => {
    const lineIndex = syncedLyrics.findIndex(
      (line, idx) =>
        currentTime >= line.time &&
        (idx === syncedLyrics.length - 1 || currentTime < syncedLyrics[idx + 1].time)
    );
    if (lineIndex !== -1) {
      setCurrentLine(lineIndex);
    }
  }, [currentTime, syncedLyrics]);

  if (!lyrics) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <Music className="w-12 h-12 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400">Letras no disponibles</p>
      </div>
    );
  }

  if (syncedLyrics.length === 0) {
    // Mostrar letras sin sincronización
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <Music className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Letra</h3>
        </div>
        <div className="prose prose-invert max-w-none">
          <pre className="text-white whitespace-pre-wrap font-sans text-sm leading-relaxed">
            {lyrics}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Letra Sincronizada</h3>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {syncedLyrics.map((line, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg transition-all ${
              idx === currentLine
                ? 'bg-purple-600/30 border border-purple-500 text-white scale-105'
                : idx < currentLine
                ? 'text-gray-400 opacity-60'
                : 'text-gray-300'
            }`}
          >
            <p className="text-sm">{line.text || '♪'}</p>
          </div>
        ))}
      </div>
    </div>
  );
}


