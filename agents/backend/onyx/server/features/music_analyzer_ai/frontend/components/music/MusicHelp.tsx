'use client';

import { useState } from 'react';
import { HelpCircle, Book, Video, MessageCircle, ExternalLink } from 'lucide-react';

export function MusicHelp() {
  const [activeSection, setActiveSection] = useState<'getting-started' | 'features' | 'troubleshooting'>('getting-started');

  const sections = {
    'getting-started': {
      title: 'Primeros Pasos',
      content: [
        'Busca canciones usando la barra de búsqueda',
        'Selecciona una canción para ver su análisis completo',
        'Explora las diferentes pestañas para descubrir funcionalidades',
        'Usa los atajos de teclado para navegación rápida',
      ],
    },
    features: {
      title: 'Funcionalidades',
      content: [
        'Análisis completo de canciones con métricas técnicas',
        'Recomendaciones inteligentes basadas en ML',
        'Comparación de múltiples canciones',
        'Generación automática de playlists',
        'Análisis temporal y de estructura',
        'Visualizador de audio en tiempo real',
      ],
    },
    troubleshooting: {
      title: 'Solución de Problemas',
      content: [
        'Si no ves resultados, verifica tu conexión a internet',
        'Asegúrate de que el backend esté ejecutándose',
        'Recarga la página si experimentas problemas de rendimiento',
        'Revisa la consola del navegador para errores',
      ],
    },
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <HelpCircle className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Ayuda</h3>
      </div>

      <div className="flex gap-2 mb-4">
        {Object.keys(sections).map((key) => (
          <button
            key={key}
            onClick={() => setActiveSection(key as any)}
            className={`px-3 py-1 rounded-lg text-sm transition-colors ${
              activeSection === key
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {sections[key as keyof typeof sections].title}
          </button>
        ))}
      </div>

      <div className="space-y-3">
        {sections[activeSection].content.map((item, idx) => (
          <div key={idx} className="flex items-start gap-3 p-3 bg-white/5 rounded-lg">
            <div className="w-6 h-6 rounded-full bg-purple-500 flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white text-xs font-bold">{idx + 1}</span>
            </div>
            <p className="text-gray-300 text-sm">{item}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-6 border-t border-white/10">
        <h4 className="text-white font-medium mb-3">Recursos Adicionales</h4>
        <div className="space-y-2">
          <a
            href="#"
            className="flex items-center gap-2 p-2 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-gray-300 hover:text-white"
          >
            <Book className="w-4 h-4" />
            <span className="text-sm">Documentación</span>
            <ExternalLink className="w-3 h-3 ml-auto" />
          </a>
          <a
            href="#"
            className="flex items-center gap-2 p-2 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-gray-300 hover:text-white"
          >
            <Video className="w-4 h-4" />
            <span className="text-sm">Tutoriales en Video</span>
            <ExternalLink className="w-3 h-3 ml-auto" />
          </a>
          <a
            href="#"
            className="flex items-center gap-2 p-2 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-gray-300 hover:text-white"
          >
            <MessageCircle className="w-4 h-4" />
            <span className="text-sm">Soporte</span>
            <ExternalLink className="w-3 h-3 ml-auto" />
          </a>
        </div>
      </div>
    </div>
  );
}


