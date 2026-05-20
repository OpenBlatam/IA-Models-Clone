'use client';

import { useState } from 'react';
import { Trophy, Medal, Award, Crown } from 'lucide-react';

interface LeaderboardEntry {
  rank: number;
  username: string;
  score: number;
  avatar?: string;
}

export function MusicLeaderboard() {
  const [leaderboard] = useState<LeaderboardEntry[]>([
    { rank: 1, username: 'MusicMaster', score: 1250, avatar: '👑' },
    { rank: 2, username: 'BeatHunter', score: 980, avatar: '🎵' },
    { rank: 3, username: 'SoundExplorer', score: 850, avatar: '🎶' },
    { rank: 4, username: 'MelodySeeker', score: 720, avatar: '🎼' },
    { rank: 5, username: 'RhythmKing', score: 650, avatar: '🥁' },
  ]);

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Crown className="w-5 h-5 text-yellow-400" />;
      case 2:
        return <Medal className="w-5 h-5 text-gray-300" />;
      case 3:
        return <Award className="w-5 h-5 text-orange-400" />;
      default:
        return <Trophy className="w-4 h-4 text-gray-500" />;
    }
  };

  const getRankColor = (rank: number) => {
    switch (rank) {
      case 1:
        return 'bg-yellow-500/20 border-yellow-500/50';
      case 2:
        return 'bg-gray-400/20 border-gray-400/50';
      case 3:
        return 'bg-orange-500/20 border-orange-500/50';
      default:
        return 'bg-white/5 border-white/10';
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Trophy className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold text-white">Clasificación</h3>
      </div>

      <div className="space-y-2">
        {leaderboard.map((entry) => (
          <div
            key={entry.rank}
            className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${getRankColor(
              entry.rank
            )}`}
          >
            <div className="flex items-center justify-center w-8">
              {getRankIcon(entry.rank)}
            </div>
            <div className="flex-1 flex items-center gap-3">
              <div className="text-2xl">{entry.avatar || '👤'}</div>
              <div className="flex-1">
                <p className="text-white font-medium text-sm">{entry.username}</p>
                <p className="text-xs text-gray-400">{entry.score} puntos</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-lg font-bold text-white">#{entry.rank}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-white/10">
        <div className="flex items-center gap-3 p-3 bg-purple-500/20 rounded-lg border border-purple-500/30">
          <div className="text-2xl">👤</div>
          <div className="flex-1">
            <p className="text-white font-medium text-sm">Tú</p>
            <p className="text-xs text-gray-400">250 puntos</p>
          </div>
          <div className="text-right">
            <span className="text-lg font-bold text-purple-300">#12</span>
          </div>
        </div>
      </div>
    </div>
  );
}


