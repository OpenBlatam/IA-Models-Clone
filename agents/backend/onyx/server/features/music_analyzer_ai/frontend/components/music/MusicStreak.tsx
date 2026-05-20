'use client';

import { useState, useEffect } from 'react';
import { Flame, Calendar } from 'lucide-react';

export function MusicStreak() {
  const [streak, setStreak] = useState(0);
  const [lastActiveDate, setLastActiveDate] = useState<string | null>(null);

  useEffect(() => {
    const savedStreak = localStorage.getItem('music-streak');
    const savedDate = localStorage.getItem('music-last-active');

    if (savedDate) {
      const lastDate = new Date(savedDate);
      const today = new Date();
      const diffTime = today.getTime() - lastDate.getTime();
      const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays === 0) {
        // Mismo día, mantener streak
        setStreak(parseInt(savedStreak || '0'));
      } else if (diffDays === 1) {
        // Día siguiente, incrementar streak
        const newStreak = parseInt(savedStreak || '0') + 1;
        setStreak(newStreak);
        localStorage.setItem('music-streak', newStreak.toString());
        localStorage.setItem('music-last-active', today.toISOString());
      } else {
        // Streak roto, reiniciar
        setStreak(1);
        localStorage.setItem('music-streak', '1');
        localStorage.setItem('music-last-active', today.toISOString());
      }
    } else {
      // Primera vez
      setStreak(1);
      localStorage.setItem('music-streak', '1');
      localStorage.setItem('music-last-active', new Date().toISOString());
    }
  }, []);

  return (
    <div className="bg-gradient-to-br from-orange-500/20 to-red-500/20 backdrop-blur-lg rounded-xl p-4 border border-orange-500/30">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/30 rounded-lg">
            <Flame className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <p className="text-xs text-gray-400">Racha de días</p>
            <p className="text-2xl font-bold text-white">{streak}</p>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
            <Calendar className="w-3 h-3" />
            <span>Última actividad: Hoy</span>
          </div>
          {streak >= 7 && (
            <span className="text-xs text-orange-400 font-semibold">🔥 En llamas!</span>
          )}
        </div>
      </div>
    </div>
  );
}


