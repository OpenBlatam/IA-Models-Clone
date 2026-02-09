'use client';

import { useState } from 'react';
import { Book, Search, FileText, Code, Image } from 'lucide-react';

interface DocSection {
  id: string;
  title: string;
  content: string;
  category: 'guide' | 'api' | 'examples' | 'troubleshooting';
}

export default function DocumentationViewer() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'guide' | 'api' | 'examples' | 'troubleshooting'>('all');
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);

  const docs: DocSection[] = [
    {
      id: '1',
      title: 'Guía de Inicio Rápido',
      content: 'Esta es una guía completa para comenzar a usar el sistema de control del robot...',
      category: 'guide',
    },
    {
      id: '2',
      title: 'API Reference',
      content: 'Documentación completa de la API REST y WebSocket...',
      category: 'api',
    },
    {
      id: '3',
      title: 'Ejemplos de Uso',
      content: 'Ejemplos prácticos de cómo usar las diferentes funcionalidades...',
      category: 'examples',
    },
    {
      id: '4',
      title: 'Solución de Problemas',
      content: 'Guía para resolver problemas comunes...',
      category: 'troubleshooting',
    },
  ];

  const filteredDocs = docs.filter((doc) => {
    const matchesSearch = doc.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      doc.content.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || doc.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const selectedDocContent = docs.find((d) => d.id === selectedDoc);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'guide':
        return <Book className="w-4 h-4" />;
      case 'api':
        return <Code className="w-4 h-4" />;
      case 'examples':
        return <FileText className="w-4 h-4" />;
      case 'troubleshooting':
        return <Image className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Book className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Documentación</h3>
        </div>

        {/* Search */}
        <div className="mb-4 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Buscar en documentación..."
            className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>

        {/* Categories */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {(['all', 'guide', 'api', 'examples', 'troubleshooting'] as const).map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                selectedCategory === cat
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {cat === 'all' ? 'Todas' : cat}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Docs List */}
          <div className="lg:col-span-1 space-y-2">
            {filteredDocs.map((doc) => (
              <button
                key={doc.id}
                onClick={() => setSelectedDoc(doc.id)}
                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                  selectedDoc === doc.id
                    ? 'border-primary-500 bg-primary-500/10'
                    : 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {getCategoryIcon(doc.category)}
                  <h4 className="font-semibold text-white text-sm">{doc.title}</h4>
                </div>
                <p className="text-xs text-gray-400 line-clamp-2">{doc.content}</p>
              </button>
            ))}
          </div>

          {/* Doc Content */}
          <div className="lg:col-span-2">
            {selectedDocContent ? (
              <div className="p-6 bg-gray-700/50 rounded-lg border border-gray-600">
                <h2 className="text-xl font-bold text-white mb-4">{selectedDocContent.title}</h2>
                <div className="prose prose-invert max-w-none">
                  <p className="text-gray-300 whitespace-pre-wrap">{selectedDocContent.content}</p>
                </div>
              </div>
            ) : (
              <div className="p-12 text-center text-gray-400">
                <Book className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Selecciona un documento para ver su contenido</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


