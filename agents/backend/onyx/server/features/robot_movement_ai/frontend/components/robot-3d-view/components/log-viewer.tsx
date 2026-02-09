/**
 * Log Viewer Component
 * @module robot-3d-view/components/log-viewer
 */

'use client';

import { memo, useState, useEffect } from 'react';
import { advancedLogger, type LogEntry, type LogLevel } from '../utils/logger-advanced';

/**
 * Log Viewer Component
 * 
 * Displays application logs with filtering and search.
 * 
 * @returns Log viewer component
 */
export const LogViewer = memo(() => {
  const [isOpen, setIsOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<LogLevel | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | 'all'>('all');

  useEffect(() => {
    if (!isOpen) return;

    const updateLogs = () => {
      const allLogs = advancedLogger.getBuffer();
      setLogs([...allLogs].reverse()); // Most recent first
    };

    updateLogs();
    const interval = setInterval(updateLogs, 500);
    return () => clearInterval(interval);
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Open with Ctrl+L or Cmd+L
      if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const filteredLogs = logs.filter((log) => {
    if (filterLevel !== 'all' && log.level !== filterLevel) return false;
    if (selectedCategory !== 'all' && log.category !== selectedCategory) return false;
    if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    return true;
  });

  const categories = Array.from(
    new Set(logs.map((log) => log.category || 'uncategorized'))
  );

  const getLevelColor = (level: LogLevel): string => {
    switch (level) {
      case 'debug':
        return 'text-gray-400';
      case 'info':
        return 'text-blue-400';
      case 'warn':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      case 'fatal':
        return 'text-red-600';
      default:
        return 'text-gray-300';
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="absolute inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={() => setIsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Log viewer"
    >
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-4 max-w-4xl w-full mx-4 max-h-[80vh] flex flex-col shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Log Viewer</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close log viewer"
          >
            ✕
          </button>
        </div>

        {/* Filters */}
        <div className="flex gap-2 mb-4 flex-wrap">
          <select
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value as LogLevel | 'all')}
            className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="all">All Levels</option>
            <option value="debug">Debug</option>
            <option value="info">Info</option>
            <option value="warn">Warn</option>
            <option value="error">Error</option>
            <option value="fatal">Fatal</option>
          </select>

          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="all">All Categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>

          <input
            type="text"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm placeholder-gray-400"
          />

          <button
            onClick={() => {
              advancedLogger.clear();
              setLogs([]);
            }}
            className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-white text-sm transition-colors"
          >
            Clear
          </button>

          <button
            onClick={() => {
              const exported = advancedLogger.export();
              const blob = new Blob([exported], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `logs-${Date.now()}.json`;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            }}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm transition-colors"
          >
            Export
          </button>
        </div>

        {/* Logs */}
        <div className="flex-1 overflow-y-auto bg-gray-900/50 rounded p-2 font-mono text-xs">
          {filteredLogs.length === 0 ? (
            <div className="text-gray-400 text-center py-8">No logs found</div>
          ) : (
            filteredLogs.map((log, index) => (
              <div
                key={index}
                className="mb-1 p-2 bg-gray-800/50 rounded hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-start gap-2">
                  <span className={`font-semibold ${getLevelColor(log.level)}`}>
                    [{log.level.toUpperCase()}]
                  </span>
                  <span className="text-gray-400">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  {log.category && (
                    <span className="text-gray-500">[{log.category}]</span>
                  )}
                  <span className="text-gray-300 flex-1">{log.message}</span>
                </div>
                {log.context && Object.keys(log.context).length > 0 && (
                  <details className="mt-1 ml-4">
                    <summary className="text-gray-400 cursor-pointer">Context</summary>
                    <pre className="mt-1 text-gray-500 text-xs overflow-x-auto">
                      {JSON.stringify(log.context, null, 2)}
                    </pre>
                  </details>
                )}
                {log.stack && (
                  <details className="mt-1 ml-4">
                    <summary className="text-gray-400 cursor-pointer">Stack</summary>
                    <pre className="mt-1 text-gray-500 text-xs overflow-x-auto">
                      {log.stack}
                    </pre>
                  </details>
                )}
              </div>
            ))
          )}
        </div>

        <div className="mt-2 text-xs text-gray-400">
          Press <kbd className="px-1 py-0.5 bg-gray-700 rounded">Ctrl+L</kbd> to open,{' '}
          <kbd className="px-1 py-0.5 bg-gray-700 rounded">Esc</kbd> to close
        </div>
      </div>
    </div>
  );
});

LogViewer.displayName = 'LogViewer';



