'use client';

import { useState } from 'react';
import { Shield, Filter, Download, Search, Calendar } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface AuditEntry {
  id: string;
  timestamp: Date;
  user: string;
  action: string;
  resource: string;
  status: 'success' | 'failed' | 'warning';
  ip?: string;
}

export default function AuditLog() {
  const [entries] = useState<AuditEntry[]>([
    {
      id: '1',
      timestamp: new Date(),
      user: 'admin@example.com',
      action: 'Login',
      resource: 'Sistema',
      status: 'success',
      ip: '192.168.1.100',
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 3600000),
      user: 'operator@example.com',
      action: 'Mover Robot',
      resource: 'Robot Control',
      status: 'success',
      ip: '192.168.1.101',
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 7200000),
      user: 'admin@example.com',
      action: 'Cambiar Configuración',
      resource: 'Settings',
      status: 'success',
      ip: '192.168.1.100',
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 10800000),
      user: 'user@example.com',
      action: 'Acceso Denegado',
      resource: 'Admin Panel',
      status: 'failed',
      ip: '192.168.1.102',
    },
  ]);
  const [filter, setFilter] = useState<'all' | 'success' | 'failed' | 'warning'>('all');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredEntries = entries.filter((entry) => {
    const matchesFilter = filter === 'all' || entry.status === filter;
    const matchesSearch =
      entry.user.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.action.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.resource.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const handleExport = () => {
    toast.success('Registro de auditoría exportado');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'bg-green-500/10 border-green-500/50 text-green-400';
      case 'failed':
        return 'bg-red-500/10 border-red-500/50 text-red-400';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/50 text-yellow-400';
      default:
        return 'bg-gray-700/50 border-gray-600 text-gray-300';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Registro de Auditoría</h3>
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
              placeholder="Buscar en auditoría..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            {(['all', 'success', 'failed', 'warning'] as const).map((f) => (
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

        {/* Entries List */}
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredEntries.map((entry) => (
            <div
              key={entry.id}
              className={`p-4 rounded-lg border ${getStatusColor(entry.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-white">{entry.action}</span>
                    <span className="text-sm text-gray-400">en</span>
                    <span className="text-sm text-white">{entry.resource}</span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-gray-300">
                    <span>Usuario: {entry.user}</span>
                    {entry.ip && <span>IP: {entry.ip}</span>}
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      {entry.timestamp.toLocaleString('es-ES')}
                    </span>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded text-xs capitalize ${
                  entry.status === 'success'
                    ? 'bg-green-500/20 text-green-400'
                    : entry.status === 'failed'
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {entry.status}
                </span>
              </div>
            </div>
          ))}
        </div>

        {filteredEntries.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Shield className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No se encontraron entradas</p>
          </div>
        )}
      </div>
    </div>
  );
}


