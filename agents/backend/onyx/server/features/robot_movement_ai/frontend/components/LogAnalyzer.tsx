'use client';

import { useState } from 'react';
import { FileSearch, Filter, Download, TrendingUp } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface LogEntry {
  id: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  timestamp: Date;
  source: string;
}

export default function LogAnalyzer() {
  const [logs] = useState<LogEntry[]>([
    {
      id: '1',
      level: 'info',
      message: 'Sistema iniciado correctamente',
      timestamp: new Date(),
      source: 'System',
    },
    {
      id: '2',
      level: 'warning',
      message: 'Velocidad alta detectada',
      timestamp: new Date(Date.now() - 60000),
      source: 'Safety',
    },
    {
      id: '3',
      level: 'error',
      message: 'Error de conexión con el robot',
      timestamp: new Date(Date.now() - 120000),
      source: 'Robot',
    },
    {
      id: '4',
      level: 'debug',
      message: 'Procesando comando de movimiento',
      timestamp: new Date(Date.now() - 180000),
      source: 'Control',
    },
  ]);
  const [filter, setFilter] = useState<'all' | 'info' | 'warning' | 'error' | 'debug'>('all');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredLogs = logs.filter((log) => {
    const matchesFilter = filter === 'all' || log.level === filter;
    const matchesSearch = log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.source.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const handleExport = () => {
    toast.success('Logs exportados');
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error':
        return 'bg-red-500/10 border-red-500/50 text-red-400';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/50 text-yellow-400';
      case 'debug':
        return 'bg-blue-500/10 border-blue-500/50 text-blue-400';
      default:
        return 'bg-green-500/10 border-green-500/50 text-green-400';
    }
  };

  const levelCounts = {
    all: logs.length,
    info: logs.filter((l) => l.level === 'info').length,
    warning: logs.filter((l) => l.level === 'warning').length,
    error: logs.filter((l) => l.level === 'error').length,
    debug: logs.filter((l) => l.level === 'debug').length,
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <FileSearch className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Analizador de Logs</h3>
          </div>
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Exportar
          </button>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-5 gap-2 mb-6">
          {(['all', 'info', 'warning', 'error', 'debug'] as const).map((level) => (
            <div
              key={level}
              className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                filter === level
                  ? 'bg-primary-600/20 border-primary-500'
                  : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
              }`}
              onClick={() => setFilter(level)}
            >
              <p className="text-xs text-gray-400 mb-1 capitalize">{level}</p>
              <p className="text-lg font-bold text-white">{levelCounts[level]}</p>
            </div>
          ))}
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <FileSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar en logs..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        {/* Logs List */}
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredLogs.map((log) => (
            <div
              key={log.id}
              className={`p-3 rounded-lg border ${getLevelColor(log.level)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-semibold uppercase">{log.level}</span>
                    <span className="text-xs text-gray-400">{log.source}</span>
                  </div>
                  <p className="text-sm text-white">{log.message}</p>
                </div>
                <span className="text-xs text-gray-400 ml-4">
                  {log.timestamp.toLocaleTimeString('es-ES')}
                </span>
              </div>
            </div>
          ))}
        </div>

        {filteredLogs.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <FileSearch className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No se encontraron logs</p>
          </div>
        )}
      </div>
    </div>
  );
}


