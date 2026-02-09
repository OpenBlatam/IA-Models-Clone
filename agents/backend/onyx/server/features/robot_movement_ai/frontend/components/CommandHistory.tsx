'use client';

import { useState, useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { History, Play, Trash2, Copy, Search } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { format } from 'date-fns';

interface CommandEntry {
  id: string;
  command: string;
  timestamp: Date;
  success: boolean;
  response?: any;
}

export default function CommandHistory() {
  const { chatMessages } = useRobotStore();
  const [commands, setCommands] = useState<CommandEntry[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSuccess, setFilterSuccess] = useState<'all' | 'success' | 'error'>('all');

  useEffect(() => {
    // Extract commands from chat messages
    const commandEntries: CommandEntry[] = chatMessages
      .filter((msg) => msg.role === 'user')
      .map((msg, index) => ({
        id: `cmd-${index}-${Date.now()}`,
        command: msg.content,
        timestamp: new Date(),
        success: true, // Would need to track this from responses
      }));
    setCommands(commandEntries);
  }, [chatMessages]);

  const filteredCommands = commands.filter((cmd) => {
    const matchesSearch = cmd.command.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter =
      filterSuccess === 'all' ||
      (filterSuccess === 'success' && cmd.success) ||
      (filterSuccess === 'error' && !cmd.success);
    return matchesSearch && matchesFilter;
  });

  const handleReplay = (command: string) => {
    // Would trigger command execution
    toast.info(`Re-ejecutando: ${command}`);
  };

  const handleCopy = (command: string) => {
    navigator.clipboard.writeText(command);
    toast.success('Comando copiado');
  };

  const handleDelete = (id: string) => {
    setCommands((prev) => prev.filter((cmd) => cmd.id !== id));
    toast.success('Comando eliminado');
  };

  const clearHistory = () => {
    setCommands([]);
    toast.success('Historial limpiado');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Historial de Comandos</h3>
          </div>
          <button
            onClick={clearHistory}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Trash2 className="w-4 h-4" />
            Limpiar
          </button>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar comandos..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <select
            value={filterSuccess}
            onChange={(e) => setFilterSuccess(e.target.value as any)}
            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="all">Todos</option>
            <option value="success">Exitosos</option>
            <option value="error">Con error</option>
          </select>
        </div>

        {/* Commands List */}
        <div className="space-y-2 max-h-[600px] overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <History className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay comandos en el historial</p>
            </div>
          ) : (
            filteredCommands.map((cmd) => (
              <div
                key={cmd.id}
                className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-gray-500 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          cmd.success ? 'bg-green-400' : 'bg-red-400'
                        }`}
                      />
                      <span className="text-xs text-gray-400">
                        {format(cmd.timestamp, 'HH:mm:ss')}
                      </span>
                    </div>
                    <p className="text-white font-mono text-sm break-all">{cmd.command}</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleReplay(cmd.command)}
                      className="p-2 bg-primary-600 hover:bg-primary-700 text-white rounded transition-colors"
                      title="Re-ejecutar"
                    >
                      <Play className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleCopy(cmd.command)}
                      className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                      title="Copiar"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(cmd.id)}
                      className="p-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                      title="Eliminar"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Stats */}
        <div className="mt-6 pt-4 border-t border-gray-700 grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-white">{commands.length}</p>
            <p className="text-xs text-gray-400">Total</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-400">
              {commands.filter((c) => c.success).length}
            </p>
            <p className="text-xs text-gray-400">Exitosos</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-400">
              {commands.filter((c) => !c.success).length}
            </p>
            <p className="text-xs text-gray-400">Con error</p>
          </div>
        </div>
      </div>
    </div>
  );
}


