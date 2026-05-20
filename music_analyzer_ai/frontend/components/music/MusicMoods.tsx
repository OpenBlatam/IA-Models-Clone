'use client';

import { useState } from 'react';
import { Heart, Zap, Moon, Sun, Coffee, Dumbbell, Book, Car } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicMoodsProps {
  onMoodSelect?: (mood: string) => void;
}

export function MusicMoods({ onMoodSelect }: MusicMoodsProps) {
  const [selectedMood, setSelectedMood] = useState<string | null>(null);

  const moods = [
    { id: 'happy', label: 'Feliz', icon: Sun, color: 'bg-yellow-500' },
    { id: 'sad', label: 'Triste', icon: Moon, color: 'bg-blue-500' },
    { id: 'energetic', label: 'Energético', icon: Zap, color: 'bg-red-500' },
    { id: 'romantic', label: 'Romántico', icon: Heart, color: 'bg-pink-500' },
    { id: 'calm', label: 'Calmado', icon: Coffee, color: 'bg-green-500' },
    { id: 'workout', label: 'Ejercicio', icon: Dumbbell, color: 'bg-orange-500' },
    { id: 'study', label: 'Estudio', icon: Book, color: 'bg-purple-500' },
    { id: 'driving', label: 'Conducción', icon: Car, color: 'bg-gray-500' },
  ];

  const handleMoodClick = (moodId: string) => {
    setSelectedMood(moodId);
    onMoodSelect?.(moodId);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h3 className="text-lg font-semibold text-white mb-4">Moods y Actividades</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {moods.map((mood) => {
          const Icon = mood.icon;
          return (
            <button
              key={mood.id}
              onClick={() => handleMoodClick(mood.id)}
              className={`p-4 rounded-lg transition-all ${
                selectedMood === mood.id
                  ? 'bg-purple-600 scale-105'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className={`w-12 h-12 ${mood.color} rounded-full flex items-center justify-center mx-auto mb-2`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
              <p className="text-sm text-white font-medium">{mood.label}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}


