'use client';

import { useState } from 'react';
import { Plug, Plus, Trash2, CheckCircle, XCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface APIIntegration {
  id: string;
  name: string;
  url: string;
  status: 'connected' | 'disconnected' | 'error';
  lastSync?: Date;
}

export default function APIIntegrations() {
  const [integrations, setIntegrations] = useState<APIIntegration[]>([
    {
      id: '1',
      name: 'API Externa 1',
      url: 'https://api.example.com',
      status: 'connected',
      lastSync: new Date(),
    },
    {
      id: '2',
      name: 'API Externa 2',
      url: 'https://api.another.com',
      status: 'disconnected',
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');

  const handleAdd = () => {
    if (!newName.trim() || !newUrl.trim()) {
      toast.error('Nombre y URL son requeridos');
      return;
    }

    const newIntegration: APIIntegration = {
      id: Date.now().toString(),
      name: newName,
      url: newUrl,
      status: 'disconnected',
    };

    setIntegrations([...integrations, newIntegration]);
    setNewName('');
    setNewUrl('');
    setShowAdd(false);
    toast.success('Integración agregada');
  };

  const handleDelete = (id: string) => {
    setIntegrations(integrations.filter((i) => i.id !== id));
    toast.success('Integración eliminada');
  };

  const handleConnect = (id: string) => {
    setIntegrations((prev) =>
      prev.map((i) =>
        i.id === id
          ? { ...i, status: 'connected' as const, lastSync: new Date() }
          : i
      )
    );
    toast.success('Conectado exitosamente');
  };

  const handleDisconnect = (id: string) => {
    setIntegrations((prev) =>
      prev.map((i) => (i.id === id ? { ...i, status: 'disconnected' as const } : i))
    );
    toast.info('Desconectado');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <XCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Plug className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Integraciones API</h3>
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Agregar
          </button>
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">Nueva Integración</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Nombre de la integración"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <input
                type="url"
                value={newUrl}
                onChange={(e) => setNewUrl(e.target.value)}
                placeholder="URL de la API"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleAdd}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  Agregar
                </button>
                <button
                  onClick={() => {
                    setShowAdd(false);
                    setNewName('');
                    setNewUrl('');
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Integrations List */}
        <div className="space-y-3">
          {integrations.map((integration) => (
            <div
              key={integration.id}
              className={`p-4 rounded-lg border ${
                integration.status === 'connected'
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(integration.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{integration.name}</h4>
                    <p className="text-sm text-gray-300 mb-1">{integration.url}</p>
                    {integration.lastSync && (
                      <p className="text-xs text-gray-400">
                        Última sincronización: {integration.lastSync.toLocaleString('es-ES')}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  {integration.status === 'connected' ? (
                    <button
                      onClick={() => handleDisconnect(integration.id)}
                      className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded-lg transition-colors"
                    >
                      Desconectar
                    </button>
                  ) : (
                    <button
                      onClick={() => handleConnect(integration.id)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
                    >
                      Conectar
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(integration.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {integrations.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Plug className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay integraciones configuradas</p>
          </div>
        )}
      </div>
    </div>
  );
}


