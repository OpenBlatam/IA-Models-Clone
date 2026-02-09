'use client';

import { useState } from 'react';
import { Database, Search, Download, Filter } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface DataTable {
  name: string;
  rows: number;
  size: string;
  lastUpdate: Date;
}

export default function DataExplorer() {
  const [tables] = useState<DataTable[]>([
    {
      name: 'movements',
      rows: 1247,
      size: '2.5 MB',
      lastUpdate: new Date(),
    },
    {
      name: 'logs',
      rows: 5432,
      size: '8.3 MB',
      lastUpdate: new Date(Date.now() - 3600000),
    },
    {
      name: 'users',
      rows: 45,
      size: '120 KB',
      lastUpdate: new Date(Date.now() - 7200000),
    },
    {
      name: 'settings',
      rows: 12,
      size: '45 KB',
      lastUpdate: new Date(Date.now() - 86400000),
    },
  ]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTable, setSelectedTable] = useState<string | null>(null);

  const filteredTables = tables.filter((table) =>
    table.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleExport = (tableName: string) => {
    toast.success(`Exportando tabla ${tableName}...`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Database className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Explorador de Datos</h3>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar tablas..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        {/* Tables List */}
        <div className="space-y-3">
          {filteredTables.map((table) => (
            <div
              key={table.name}
              className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                selectedTable === table.name
                  ? 'bg-primary-600/20 border-primary-500'
                  : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
              }`}
              onClick={() => setSelectedTable(table.name)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Database className="w-4 h-4 text-primary-400" />
                    <h4 className="font-semibold text-white">{table.name}</h4>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-300">
                    <span>{table.rows.toLocaleString()} filas</span>
                    <span>{table.size}</span>
                    <span className="text-xs text-gray-400">
                      Actualizado: {table.lastUpdate.toLocaleString('es-ES')}
                    </span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleExport(table.name);
                  }}
                  className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                >
                  <Download className="w-3 h-3" />
                  Exportar
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Table Preview */}
        {selectedTable && (
          <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">
              Vista previa: {selectedTable}
            </h4>
            <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="text-left text-gray-300 p-2">ID</th>
                    <th className="text-left text-gray-300 p-2">Datos</th>
                    <th className="text-left text-gray-300 p-2">Fecha</th>
                  </tr>
                </thead>
                <tbody>
                  {[1, 2, 3, 4, 5].map((i) => (
                    <tr key={i} className="border-b border-gray-700">
                      <td className="text-white p-2">{i}</td>
                      <td className="text-gray-300 p-2">Datos de ejemplo {i}</td>
                      <td className="text-gray-400 p-2">
                        {new Date().toLocaleDateString('es-ES')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


