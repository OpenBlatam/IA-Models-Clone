'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api/client';
import { Terminal, Trash2, Download, Filter } from 'lucide-react';
import { format } from 'date-fns';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
  source?: string;
}

export default function LogsPanel() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<'all' | 'info' | 'warning' | 'error' | 'success'>('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Simular logs en tiempo real (en producción vendría del backend)
    const interval = setInterval(() => {
      const newLog: LogEntry = {
        timestamp: new Date().toISOString(),
        level: ['info', 'warning', 'error', 'success'][Math.floor(Math.random() * 4)] as any,
        message: `Log entry ${logs.length + 1}`,
        source: 'robot-system',
      };
      setLogs((prev) => [...prev.slice(-99), newLog]);
    }, 2000);

    return () => clearInterval(interval);
  }, [logs.length]);

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter(
    (log) => filter === 'all' || log.level === filter
  );

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error':
        return 'text-red-400 bg-red-500/20';
      case 'warning':
        return 'text-yellow-400 bg-yellow-500/20';
      case 'success':
        return 'text-green-400 bg-green-500/20';
      default:
        return 'text-blue-400 bg-blue-500/20';
    }
  };

  const handleClear = () => {
    setLogs([]);
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(logs, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `robot-logs-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 flex flex-col h-[600px]">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Logs del Sistema</h3>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="all">Todos</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
            <option value="success">Success</option>
          </select>
          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded"
            />
            Auto-scroll
          </label>
          <button
            onClick={handleExport}
            className="p-2 hover:bg-gray-700 rounded transition-colors"
            title="Exportar logs"
          >
            <Download className="w-4 h-4 text-gray-300" />
          </button>
          <button
            onClick={handleClear}
            className="p-2 hover:bg-gray-700 rounded transition-colors"
            title="Limpiar logs"
          >
            <Trash2 className="w-4 h-4 text-gray-300" />
          </button>
        </div>
      </div>

      {/* Logs */}
      <div className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-1">
        {filteredLogs.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">No hay logs disponibles</div>
        ) : (
          filteredLogs.map((log, index) => (
            <div
              key={index}
              className={`p-2 rounded ${getLevelColor(log.level)}`}
            >
              <div className="flex items-start gap-2">
                <span className="text-xs opacity-70">
                  {format(new Date(log.timestamp), 'HH:mm:ss.SSS')}
                </span>
                <span className="font-semibold uppercase text-xs">{log.level}</span>
                {log.source && (
                  <span className="text-xs opacity-70">[{log.source}]</span>
                )}
                <span className="flex-1">{log.message}</span>
              </div>
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>

      {/* Footer */}
      <div className="p-2 border-t border-gray-700 text-xs text-gray-400 text-center">
        {logs.length} logs totales | {filteredLogs.length} mostrados
      </div>
    </div>
  );
}

