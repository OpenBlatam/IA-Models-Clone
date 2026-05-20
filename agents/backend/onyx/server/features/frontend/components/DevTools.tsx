'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiTerminal, FiX, FiTrash2, FiDownload } from 'react-icons/fi';

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  data?: any;
}

export default function DevTools() {
  const [isOpen, setIsOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<'all' | 'info' | 'warn' | 'error' | 'debug'>('all');

  useEffect(() => {
    // Intercept console methods
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;

    console.log = (...args) => {
      addLog('info', args.join(' '), args);
      originalLog(...args);
    };

    console.warn = (...args) => {
      addLog('warn', args.join(' '), args);
      originalWarn(...args);
    };

    console.error = (...args) => {
      addLog('error', args.join(' '), args);
      originalError(...args);
    };

    return () => {
      console.log = originalLog;
      console.warn = originalWarn;
      console.error = originalError;
    };
  }, []);

  const addLog = (level: LogEntry['level'], message: string, data?: any) => {
    const entry: LogEntry = {
      id: Date.now().toString() + Math.random(),
      timestamp: new Date(),
      level,
      message,
      data,
    };
    setLogs((prev) => [...prev, entry].slice(-100)); // Keep last 100
  };

  const clearLogs = () => {
    setLogs([]);
  };

  const exportLogs = () => {
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bul-logs-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const filteredLogs = filter === 'all' ? logs : logs.filter((log) => log.level === filter);

  // Toggle with Ctrl+Shift+D
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        setIsOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: '100%' }}
        animate={{ y: 0 }}
        exit={{ y: '100%' }}
        className="fixed bottom-0 left-0 right-0 h-96 bg-gray-900 text-white z-50 border-t border-gray-700 flex flex-col"
      >
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FiTerminal size={20} />
            <h3 className="font-semibold">DevTools</h3>
            <div className="flex gap-2">
              {(['all', 'info', 'warn', 'error', 'debug'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setFilter(level)}
                  className={`px-2 py-1 text-xs rounded ${
                    filter === level
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={exportLogs} className="btn-icon text-white hover:bg-gray-700">
              <FiDownload size={16} />
            </button>
            <button onClick={clearLogs} className="btn-icon text-white hover:bg-gray-700">
              <FiTrash2 size={16} />
            </button>
            <button onClick={() => setIsOpen(false)} className="btn-icon text-white hover:bg-gray-700">
              <FiX size={20} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 font-mono text-xs">
          {filteredLogs.length === 0 ? (
            <div className="text-gray-500 text-center py-8">No hay logs</div>
          ) : (
            filteredLogs.map((log) => (
              <div
                key={log.id}
                className={`mb-2 p-2 rounded ${
                  log.level === 'error'
                    ? 'bg-red-900/30 text-red-300'
                    : log.level === 'warn'
                    ? 'bg-yellow-900/30 text-yellow-300'
                    : log.level === 'debug'
                    ? 'bg-blue-900/30 text-blue-300'
                    : 'bg-gray-800 text-gray-300'
                }`}
              >
                <div className="flex items-start gap-2">
                  <span className="text-gray-500">
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                  <span className="font-bold uppercase">{log.level}</span>
                  <span className="flex-1">{log.message}</span>
                </div>
                {log.data && (
                  <pre className="mt-1 text-xs opacity-75 overflow-x-auto">
                    {JSON.stringify(log.data, null, 2)}
                  </pre>
                )}
              </div>
            ))
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


