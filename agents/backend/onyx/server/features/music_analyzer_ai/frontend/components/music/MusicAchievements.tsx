'use client';

import { useState } from 'react';
import { Trophy, Award, Star, Target, Zap } from 'lucide-react';

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: any;
  unlocked: boolean;
  progress: number;
  maxProgress: number;
}

export function MusicAchievements() {
  const [achievements] = useState<Achievement[]>([
    {
      id: '1',
      title: 'Primer Análisis',
      description: 'Analiza tu primera canción',
      icon: Star,
      unlocked: true,
      progress: 1,
      maxProgress: 1,
    },
    {
      id: '2',
      title: 'Coleccionista',
      description: 'Agrega 10 canciones a favoritos',
      icon: Trophy,
      unlocked: false,
      progress: 7,
      maxProgress: 10,
    },
    {
      id: '3',
      title: 'Comparador',
      description: 'Compara 5 canciones',
      icon: Target,
      unlocked: false,
      progress: 3,
      maxProgress: 5,
    },
    {
      id: '4',
      title: 'Explorador',
      description: 'Descubre 20 canciones nuevas',
      icon: Zap,
      unlocked: false,
      progress: 12,
      maxProgress: 20,
    },
    {
      id: '5',
      title: 'Maestro',
      description: 'Crea 5 playlists',
      icon: Award,
      unlocked: false,
      progress: 2,
      maxProgress: 5,
    },
  ]);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Trophy className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold text-white">Logros</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {achievements.map((achievement) => {
          const Icon = achievement.icon;
          const progressPercentage = (achievement.progress / achievement.maxProgress) * 100;

          return (
            <div
              key={achievement.id}
              className={`p-4 rounded-lg border transition-all ${
                achievement.unlocked
                  ? 'bg-yellow-500/20 border-yellow-500/50'
                  : 'bg-white/5 border-white/10'
              }`}
            >
              <div className="flex items-start gap-3">
                <div
                  className={`p-2 rounded-lg ${
                    achievement.unlocked ? 'bg-yellow-500/30' : 'bg-white/10'
                  }`}
                >
                  <Icon
                    className={`w-5 h-5 ${
                      achievement.unlocked ? 'text-yellow-400' : 'text-gray-400'
                    }`}
                  />
                </div>
                <div className="flex-1">
                  <h4
                    className={`font-semibold mb-1 ${
                      achievement.unlocked ? 'text-yellow-400' : 'text-white'
                    }`}
                  >
                    {achievement.title}
                  </h4>
                  <p className="text-xs text-gray-400 mb-2">{achievement.description}</p>
                  {!achievement.unlocked && (
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-400">Progreso</span>
                        <span className="text-gray-300">
                          {achievement.progress} / {achievement.maxProgress}
                        </span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-purple-500 h-1.5 rounded-full transition-all"
                          style={{ width: `${progressPercentage}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {achievement.unlocked && (
                    <span className="text-xs text-yellow-400 font-semibold">✓ Desbloqueado</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


