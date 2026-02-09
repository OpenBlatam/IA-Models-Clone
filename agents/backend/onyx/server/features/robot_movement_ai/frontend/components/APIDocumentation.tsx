'use client';

import { useState } from 'react';
import { Book, Search, Code, Copy } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface APIEndpoint {
  id: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  parameters?: { name: string; type: string; required: boolean }[];
}

export default function APIDocumentation() {
  const [endpoints] = useState<APIEndpoint[]>([
    {
      id: '1',
      method: 'GET',
      path: '/api/robot/status',
      description: 'Obtiene el estado actual del robot',
    },
    {
      id: '2',
      method: 'POST',
      path: '/api/robot/move',
      description: 'Mueve el robot a una posición específica',
      parameters: [
        { name: 'x', type: 'number', required: true },
        { name: 'y', type: 'number', required: true },
        { name: 'z', type: 'number', required: true },
      ],
    },
    {
      id: '3',
      method: 'GET',
      path: '/api/robot/metrics',
      description: 'Obtiene las métricas del robot',
    },
    {
      id: '4',
      method: 'POST',
      path: '/api/robot/stop',
      description: 'Detiene el movimiento del robot',
    },
  ]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | null>(null);

  const filteredEndpoints = endpoints.filter((endpoint) =>
    endpoint.path.toLowerCase().includes(searchTerm.toLowerCase()) ||
    endpoint.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCopyCode = (endpoint: APIEndpoint) => {
    const code = `curl -X ${endpoint.method} ${endpoint.path}`;
    navigator.clipboard.writeText(code);
    toast.success('Código copiado');
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET':
        return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'POST':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'PUT':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'DELETE':
        return 'bg-red-500/20 text-red-400 border-red-500/50';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Book className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Documentación de API</h3>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar endpoints..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        {/* Endpoints List */}
        <div className="space-y-3">
          {filteredEndpoints.map((endpoint) => (
            <div
              key={endpoint.id}
              className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                selectedEndpoint === endpoint.id
                  ? 'bg-primary-600/20 border-primary-500'
                  : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
              }`}
              onClick={() => setSelectedEndpoint(endpoint.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${getMethodColor(endpoint.method)}`}>
                      {endpoint.method}
                    </span>
                    <code className="text-sm text-white font-mono">{endpoint.path}</code>
                  </div>
                  <p className="text-sm text-gray-300">{endpoint.description}</p>
                  {endpoint.parameters && endpoint.parameters.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-gray-400 mb-1">Parámetros:</p>
                      <div className="space-y-1">
                        {endpoint.parameters.map((param, i) => (
                          <div key={i} className="text-xs text-gray-300">
                            <code className="text-primary-400">{param.name}</code>
                            <span className="text-gray-500"> ({param.type})</span>
                            {param.required && (
                              <span className="text-red-400 ml-1">*</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCopyCode(endpoint);
                  }}
                  className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                >
                  <Copy className="w-3 h-3" />
                  Copiar
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


