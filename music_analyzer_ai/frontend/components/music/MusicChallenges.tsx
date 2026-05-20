'use client';

import { useState } from 'react';
import { Target, CheckCircle, Circle, Clock } from 'lucide-react';

interface Challenge {
  id: string;
  title: string;
  description: string;
  reward: string;
  completed: boolean;
  deadline?: string;
}

export function MusicChallenges() {
  const [challenges] = useState<Challenge[]>([
    {
      id: '1',
      title: 'Explorador Semanal',
      description: 'Descubre 10 canciones nuevas esta semana',
      reward: '100 puntos',
      completed: false,
      deadline: 'En 3 días',
    },
    {
      id: '2',
      title: 'Analista Profesional',
      description: 'Analiza 5 canciones diferentes',
      reward: '50 puntos',
      completed: true,
    },
    {
      id: '3',
      title: 'Comparador Experto',
      description: 'Compara 3 canciones del mismo género',
      reward: '75 puntos',
      completed: false,
      deadline: 'En 5 días',
    },
    {
      id: '4',
      title: 'Creador de Playlists',
      description: 'Crea una playlist con 15 canciones',
      reward: '80 puntos',
      completed: false,
      deadline: 'En 7 días',
    },
  ]);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Target className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Desafíos</h3>
      </div>

      <div className="space-y-3">
        {challenges.map((challenge) => (
          <div
            key={challenge.id}
            className={`p-4 rounded-lg border transition-all ${
              challenge.completed
                ? 'bg-green-500/20 border-green-500/50'
                : 'bg-white/5 border-white/10 hover:bg-white/10'
            }`}
          >
            <div className="flex items-start gap-3">
              <div className="mt-1">
                {challenge.completed ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <Circle className="w-5 h-5 text-gray-400" />
                )}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <h4
                    className={`font-semibold ${
                      challenge.completed ? 'text-green-400' : 'text-white'
                    }`}
                  >
                    {challenge.title}
                  </h4>
                  {challenge.completed && (
                    <span className="text-xs text-green-400 font-semibold">✓ Completado</span>
                  )}
                </div>
                <p className="text-sm text-gray-400 mb-2">{challenge.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-purple-300 font-semibold">
                    Recompensa: {challenge.reward}
                  </span>
                  {challenge.deadline && !challenge.completed && (
                    <div className="flex items-center gap-1 text-xs text-gray-400">
                      <Clock className="w-3 h-3" />
                      <span>{challenge.deadline}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


