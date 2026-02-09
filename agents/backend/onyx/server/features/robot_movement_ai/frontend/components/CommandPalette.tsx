'use client';

import { useState, useEffect } from 'react';
import { Command, Search, ArrowRight } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface CommandItem {
  id: string;
  name: string;
  description: string;
  category: string;
  action: () => void;
}

export default function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const commands: CommandItem[] = [
    {
      id: '1',
      name: 'Mover Robot',
      description: 'Mover el robot a una posición',
      category: 'Control',
      action: () => toast.info('Ejecutando: Mover Robot'),
    },
    {
      id: '2',
      name: 'Detener Robot',
      description: 'Detener el movimiento del robot',
      category: 'Control',
      action: () => toast.info('Ejecutando: Detener Robot'),
    },
    {
      id: '3',
      name: 'Ir a Home',
      description: 'Mover el robot a la posición inicial',
      category: 'Control',
      action: () => toast.info('Ejecutando: Ir a Home'),
    },
    {
      id: '4',
      name: 'Abrir Configuración',
      description: 'Abrir el panel de configuración',
      category: 'Sistema',
      action: () => toast.info('Ejecutando: Abrir Configuración'),
    },
    {
      id: '5',
      name: 'Ver Métricas',
      description: 'Mostrar las métricas del sistema',
      category: 'Análisis',
      action: () => toast.info('Ejecutando: Ver Métricas'),
    },
  ];

  const filteredCommands = commands.filter((cmd) =>
    cmd.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    cmd.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
      if (e.key === 'Escape') {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleExecute = (command: CommandItem) => {
    command.action();
    setIsOpen(false);
    setSearchTerm('');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-start justify-center pt-32">
      <div className="w-full max-w-2xl mx-4 bg-gray-800 rounded-lg border border-gray-700 shadow-2xl">
        <div className="p-4 border-b border-gray-700">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setSelectedIndex(0);
              }}
              placeholder="Buscar comandos... (Ctrl+K)"
              className="w-full pl-10 pr-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              autoFocus
            />
          </div>
        </div>
        <div className="max-h-96 overflow-y-auto">
          {filteredCommands.length > 0 ? (
            <div className="p-2">
              {filteredCommands.map((command, index) => (
                <div
                  key={command.id}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    index === selectedIndex
                      ? 'bg-primary-600/20 border border-primary-500'
                      : 'hover:bg-gray-700'
                  }`}
                  onClick={() => handleExecute(command)}
                  onMouseEnter={() => setSelectedIndex(index)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-white">{command.name}</h4>
                        <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded">
                          {command.category}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">{command.description}</p>
                    </div>
                    <ArrowRight className="w-4 h-4 text-gray-400" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-gray-400">
              <Command className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No se encontraron comandos</p>
            </div>
          )}
        </div>
        <div className="p-3 border-t border-gray-700 text-xs text-gray-400 flex items-center justify-between">
          <span>Presiona Enter para ejecutar</span>
          <span>Esc para cerrar</span>
        </div>
      </div>
    </div>
  );
}


