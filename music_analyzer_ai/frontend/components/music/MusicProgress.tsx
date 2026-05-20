'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, BarChart3, Target, Music } from 'lucide-react';

export function MusicProgress() {
  const [stats] = useState({
    totalTracks: 150,
    totalFavorites: 45,
    totalPlaylists: 12,
    totalAnalysis: 89,
    weeklyGoal: 20,
    weeklyProgress: 15,
  });

  const weeklyProgressPercentage = (stats.weeklyProgress / stats.weeklyGoal) * 100;

  const milestones = [
    { label: '50 canciones', value: 50, current: stats.totalTracks, icon: Music as any },
    { label: '100 canciones', value: 100, current: stats.totalTracks, icon: TrendingUp },
    { label: '200 canciones', value: 200, current: stats.totalTracks, icon: Target },
  ];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Progreso</h3>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Meta Semanal</span>
            <span className="text-sm text-white font-semibold">
              {stats.weeklyProgress} / {stats.weeklyGoal}
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all"
              style={{ width: `${Math.min(weeklyProgressPercentage, 100)}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-1">
            {stats.weeklyGoal - stats.weeklyProgress} canciones restantes
          </p>
        </div>

        <div className="pt-4 border-t border-white/10">
          <h4 className="text-sm font-semibold text-white mb-3">Hitos</h4>
          <div className="space-y-2">
            {milestones.map((milestone, idx) => {
              const Icon = milestone.icon;
              const progress = Math.min((milestone.current / milestone.value) * 100, 100);
              const completed = milestone.current >= milestone.value;

              return (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Icon
                        className={`w-4 h-4 ${
                          completed ? 'text-green-400' : 'text-gray-400'
                        }`}
                      />
                      <span className="text-sm text-gray-300">{milestone.label}</span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {milestone.current} / {milestone.value}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        completed ? 'bg-green-500' : 'bg-purple-500'
                      }`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

