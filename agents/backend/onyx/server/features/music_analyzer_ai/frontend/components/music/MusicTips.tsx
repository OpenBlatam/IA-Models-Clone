'use client';

import { useState, useEffect } from 'react';
import { Lightbulb, X, ChevronRight } from 'lucide-react';

interface Tip {
  title: string;
  content: string;
  category: string;
}

export function MusicTips() {
  const [isOpen, setIsOpen] = useState(false);
  const [currentTip, setCurrentTip] = useState(0);

  const tips: Tip[] = [
    {
      title: 'Búsqueda Rápida',
      content: 'Usa Ctrl+K para abrir la búsqueda rápida desde cualquier lugar',
      category: 'Atajos',
    },
    {
      title: 'Comparación Múltiple',
      content: 'Selecciona múltiples canciones y compáralas en el tab de Comparación',
      category: 'Funcionalidad',
    },
    {
      title: 'Exportar Análisis',
      content: 'Exporta tus análisis en múltiples formatos: JSON, CSV, Markdown',
      category: 'Exportación',
    },
    {
      title: 'Playlists Inteligentes',
      content: 'Crea playlists automáticas basadas en mood, actividad o género',
      category: 'Playlists',
    },
    {
      title: 'Recomendaciones Contextuales',
      content: 'Obtén recomendaciones basadas en la hora del día, actividad o mood',
      category: 'Recomendaciones',
    },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      if (isOpen) {
        setCurrentTip((prev) => (prev + 1) % tips.length);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isOpen, tips.length]);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="p-2 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 rounded-lg transition-colors border border-yellow-500/30"
        title="Tips y consejos"
      >
        <Lightbulb className="w-5 h-5" />
      </button>
    );
  }

  const tip = tips[currentTip];

  return (
    <div className="bg-yellow-500/20 backdrop-blur-lg rounded-xl p-4 border border-yellow-500/30">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-yellow-400" />
          <span className="text-xs text-yellow-400 font-semibold">TIP</span>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="p-1 hover:bg-yellow-500/20 rounded transition-colors"
        >
          <X className="w-3 h-3 text-yellow-400" />
        </button>
      </div>
      <div className="mb-2">
        <span className="text-xs text-yellow-300/70">{tip.category}</span>
        <h4 className="text-white font-semibold text-sm mt-1">{tip.title}</h4>
        <p className="text-gray-300 text-xs mt-1">{tip.content}</p>
      </div>
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {tips.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentTip(idx)}
              className={`w-1.5 h-1.5 rounded-full transition-colors ${
                idx === currentTip ? 'bg-yellow-400' : 'bg-yellow-400/30'
              }`}
            />
          ))}
        </div>
        <button
          onClick={() => setCurrentTip((currentTip + 1) % tips.length)}
          className="p-1 hover:bg-yellow-500/20 rounded transition-colors"
        >
          <ChevronRight className="w-3 h-3 text-yellow-400" />
        </button>
      </div>
    </div>
  );
}


