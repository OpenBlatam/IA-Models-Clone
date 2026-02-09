'use client';

import { useState } from 'react';
import { GraduationCap, BookOpen, Video, FileText, CheckCircle } from 'lucide-react';

interface LearningResource {
  id: string;
  title: string;
  type: 'tutorial' | 'video' | 'documentation' | 'example';
  description: string;
  duration?: string;
  completed?: boolean;
}

export default function LearningCenter() {
  const [resources, setResources] = useState<LearningResource[]>([
    {
      id: '1',
      title: 'Introducción al Control del Robot',
      type: 'tutorial',
      description: 'Aprende los conceptos básicos de control del robot',
      duration: '15 min',
      completed: false,
    },
    {
      id: '2',
      title: 'Visualización 3D Avanzada',
      type: 'video',
      description: 'Video tutorial sobre visualización 3D',
      duration: '20 min',
      completed: true,
    },
    {
      id: '3',
      title: 'Guía de Optimización',
      type: 'documentation',
      description: 'Documentación completa sobre optimización',
      duration: '30 min',
      completed: false,
    },
    {
      id: '4',
      title: 'Ejemplo: Movimiento Circular',
      type: 'example',
      description: 'Ejemplo práctico de movimiento circular',
      duration: '10 min',
      completed: false,
    },
  ]);
  const [filter, setFilter] = useState<'all' | 'tutorial' | 'video' | 'documentation' | 'example'>('all');

  const filteredResources = resources.filter(
    (r) => filter === 'all' || r.type === filter
  );

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'tutorial':
        return <BookOpen className="w-5 h-5 text-blue-400" />;
      case 'video':
        return <Video className="w-5 h-5 text-red-400" />;
      case 'documentation':
        return <FileText className="w-5 h-5 text-green-400" />;
      case 'example':
        return <CheckCircle className="w-5 h-5 text-yellow-400" />;
      default:
        return <BookOpen className="w-5 h-5 text-gray-400" />;
    }
  };

  const handleComplete = (id: string) => {
    setResources((prev) =>
      prev.map((r) => (r.id === id ? { ...r, completed: !r.completed } : r))
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <GraduationCap className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Centro de Aprendizaje</h3>
        </div>

        {/* Filters */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {(['all', 'tutorial', 'video', 'documentation', 'example'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                filter === f
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {f === 'all' ? 'Todos' : f}
            </button>
          ))}
        </div>

        {/* Resources Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredResources.map((resource) => (
            <div
              key={resource.id}
              className={`p-4 rounded-lg border ${
                resource.completed
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start gap-3">
                {getTypeIcon(resource.type)}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{resource.title}</h4>
                    {resource.completed && (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    )}
                  </div>
                  <p className="text-sm text-gray-300 mb-2">{resource.description}</p>
                  {resource.duration && (
                    <p className="text-xs text-gray-400 mb-3">Duración: {resource.duration}</p>
                  )}
                  <div className="flex gap-2">
                    <button className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded transition-colors">
                      Comenzar
                    </button>
                    <button
                      onClick={() => handleComplete(resource.id)}
                      className={`px-3 py-1 text-sm rounded transition-colors ${
                        resource.completed
                          ? 'bg-gray-600 hover:bg-gray-700 text-white'
                          : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                      }`}
                    >
                      {resource.completed ? 'Marcar incompleto' : 'Marcar completo'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredResources.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <GraduationCap className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay recursos disponibles</p>
          </div>
        )}
      </div>
    </div>
  );
}


