'use client';

import { useState, useEffect } from 'react';
import { Sparkles, X, Check } from 'lucide-react';

interface WelcomeFeature {
  title: string;
  description: string;
}

export function MusicWelcome() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem('music-welcome-seen');
    if (!hasSeenWelcome) {
      setIsVisible(true);
    }
  }, []);

  const features: WelcomeFeature[] = [
    {
      title: 'Análisis Inteligente',
      description: 'Obtén insights detallados sobre cualquier canción',
    },
    {
      title: 'Recomendaciones Personalizadas',
      description: 'Descubre música basada en tus preferencias',
    },
    {
      title: 'Comparación Avanzada',
      description: 'Compara canciones lado a lado',
    },
    {
      title: 'Gestión Completa',
      description: 'Organiza tus favoritos, playlists y más',
    },
  ];

  const handleClose = () => {
    setIsVisible(false);
    localStorage.setItem('music-welcome-seen', 'true');
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-purple-900 via-pink-900 to-purple-900 rounded-xl p-8 max-w-2xl w-full border border-white/20 shadow-2xl">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Sparkles className="w-8 h-8 text-purple-300" />
            <h2 className="text-3xl font-bold text-white">¡Bienvenido a Music Analyzer AI!</h2>
          </div>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        </div>

        <p className="text-gray-300 mb-6 text-lg">
          Explora y analiza música con inteligencia artificial. Descubre nuevas canciones, obtén
          insights detallados y gestiona tu biblioteca musical.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 p-4 bg-white/10 rounded-lg border border-white/20"
            >
              <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-white font-semibold mb-1">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={handleClose}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors font-semibold"
        >
          Comenzar
        </button>
      </div>
    </div>
  );
}


