'use client';

import { useState } from 'react';
import { Keyboard, Search } from 'lucide-react';

interface Shortcut {
  category: string;
  shortcuts: Array<{ keys: string[]; description: string }>;
}

export default function ShortcutsGuide() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const shortcuts: Shortcut[] = [
    {
      category: 'Navegación',
      shortcuts: [
        { keys: ['Ctrl', 'K'], description: 'Búsqueda global' },
        { keys: ['Ctrl', '1'], description: 'Ir a Control' },
        { keys: ['Ctrl', '2'], description: 'Ir a Chat' },
        { keys: ['Ctrl', '3'], description: 'Ir a 3D View' },
        { keys: ['Esc'], description: 'Cerrar modales' },
      ],
    },
    {
      category: 'Control del Robot',
      shortcuts: [
        { keys: ['Space'], description: 'Detener robot' },
        { keys: ['H'], description: 'Ir a Home' },
        { keys: ['Arrow Up'], description: 'Mover adelante' },
        { keys: ['Arrow Down'], description: 'Mover atrás' },
        { keys: ['Arrow Left'], description: 'Mover izquierda' },
        { keys: ['Arrow Right'], description: 'Mover derecha' },
      ],
    },
    {
      category: 'Grabación',
      shortcuts: [
        { keys: ['Ctrl', 'R'], description: 'Iniciar/Detener grabación' },
        { keys: ['Ctrl', 'P'], description: 'Reproducir grabación' },
        { keys: ['Ctrl', 'S'], description: 'Guardar grabación' },
      ],
    },
    {
      category: 'General',
      shortcuts: [
        { keys: ['Ctrl', '?'], description: 'Mostrar esta guía' },
        { keys: ['Ctrl', 'D'], description: 'Modo oscuro/claro' },
        { keys: ['Ctrl', 'E'], description: 'Exportar datos' },
        { keys: ['F11'], description: 'Pantalla completa' },
      ],
    },
  ];

  const filteredShortcuts = shortcuts
    .map((cat) => ({
      ...cat,
      shortcuts: cat.shortcuts.filter(
        (s) =>
          s.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
          s.keys.some((k) => k.toLowerCase().includes(searchTerm.toLowerCase()))
      ),
    }))
    .filter((cat) => cat.shortcuts.length > 0);

  const categories = ['all', ...shortcuts.map((s) => s.category)];

  const displayShortcuts =
    selectedCategory === 'all'
      ? filteredShortcuts
      : filteredShortcuts.filter((cat) => cat.category === selectedCategory);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-2 mb-6">
          <Keyboard className="w-5 h-5 text-tesla-blue" />
          <h3 className="text-lg font-semibold text-tesla-black">Guía de Atajos de Teclado</h3>
        </div>

        {/* Search */}
        <div className="mb-6 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-tesla-gray-dark" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Buscar atajos..."
            className="w-full pl-10 pr-4 py-3 bg-white border border-gray-300 rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
            aria-label="Buscar atajos de teclado"
          />
        </div>

        {/* Categories */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all min-h-[44px] ${
                selectedCategory === cat
                  ? 'bg-tesla-blue text-white shadow-sm'
                  : 'bg-white border-2 border-gray-300 text-tesla-black hover:border-gray-400'
              }`}
            >
              {cat === 'all' ? 'Todas' : cat}
            </button>
          ))}
        </div>

        {/* Shortcuts List */}
        <div className="space-y-6">
          {displayShortcuts.length === 0 ? (
            <div className="text-center py-12 text-tesla-gray-dark">
              <Keyboard className="w-12 h-12 mx-auto mb-4 text-tesla-gray-light opacity-50" />
              <p className="font-medium">No se encontraron atajos</p>
            </div>
          ) : (
            displayShortcuts.map((category) => (
              <div key={category.category}>
                <h4 className="text-sm font-semibold text-tesla-gray-dark mb-4 uppercase tracking-wider">
                  {category.category}
                </h4>
                <div className="space-y-3">
                  {category.shortcuts.map((shortcut, index) => (
                    <div
                      key={index}
                      className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 p-4 bg-gray-50 rounded-md border border-gray-200 hover:shadow-sm transition-all"
                    >
                      <span className="text-tesla-black font-medium">{shortcut.description}</span>
                      <div className="flex gap-1.5 flex-wrap">
                        {shortcut.keys.map((key, keyIndex) => (
                          <span key={keyIndex} className="flex items-center gap-1.5">
                            <kbd className="px-3 py-1.5 bg-white border border-gray-300 rounded text-xs text-tesla-black font-mono font-semibold shadow-sm">
                              {key}
                            </kbd>
                            {keyIndex < shortcut.keys.length - 1 && (
                              <span className="text-tesla-gray-dark font-medium">+</span>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}


