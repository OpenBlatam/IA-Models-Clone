'use client';

import { useState } from 'react';
import { FileText, Filter, Download, Search } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Event {
  id: string;
  timestamp: Date;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  source: string;
}

export default function EventLog() {
  const [events] = useState<Event[]>([
    {
      id: '1',
      timestamp: new Date(),
      type: 'success',
      message: 'Robot movido a posición (0.5, 0.3, 0.2)',
      source: 'Control',
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 60000),
      type: 'info',
      message: 'Conexión establecida con el robot',
      source: 'Sistema',
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 120000),
      type: 'warning',
      message: 'Velocidad alta detectada',
      source: 'Seguridad',
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 180000),
      type: 'error',
      message: 'Error al conectar con API externa',
      source: 'Integración',
    },
  ]);
  const [filter, setFilter] = useState<'all' | 'info' | 'warning' | 'error' | 'success'>('all');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredEvents = events.filter((event) => {
    const matchesFilter = filter === 'all' || event.type === filter;
    const matchesSearch = event.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      event.source.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const handleExport = () => {
    toast.success('Eventos exportados');
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'text-green-400 bg-green-500/10 border-green-500/50';
      case 'error':
        return 'text-red-400 bg-red-500/10 border-red-500/50';
      case 'warning':
        return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/50';
      default:
        return 'text-blue-400 bg-blue-500/10 border-blue-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Registro de Eventos</h3>
          </div>
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Exportar
          </button>
        </div>

        {/* Filters */}
        <div className="mb-6 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar eventos..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            {(['all', 'info', 'warning', 'error', 'success'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 rounded-lg text-sm transition-colors flex items-center gap-2 ${
                  filter === f
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <Filter className="w-3 h-3" />
                {f === 'all' ? 'Todos' : f}
              </button>
            ))}
          </div>
        </div>

        {/* Events List */}
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredEvents.map((event) => (
            <div
              key={event.id}
              className={`p-3 rounded-lg border ${getTypeColor(event.type)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-semibold uppercase">{event.type}</span>
                    <span className="text-xs text-gray-400">{event.source}</span>
                  </div>
                  <p className="text-sm text-white">{event.message}</p>
                </div>
                <span className="text-xs text-gray-400 ml-4">
                  {event.timestamp.toLocaleTimeString('es-ES')}
                </span>
              </div>
            </div>
          ))}
        </div>

        {filteredEvents.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No se encontraron eventos</p>
          </div>
        )}
      </div>
    </div>
  );
}


