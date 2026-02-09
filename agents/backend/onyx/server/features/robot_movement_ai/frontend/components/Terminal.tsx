'use client';

import { useState, useRef, useEffect } from 'react';
import { Terminal as TerminalIcon, Send, Trash2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface TerminalLine {
  id: string;
  type: 'input' | 'output' | 'error';
  content: string;
  timestamp: Date;
}

export default function Terminal() {
  const [lines, setLines] = useState<TerminalLine[]>([
    {
      id: '1',
      type: 'output',
      content: 'Robot Movement AI Terminal v1.0.0',
      timestamp: new Date(),
    },
    {
      id: '2',
      type: 'output',
      content: 'Type "help" for available commands',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  const executeCommand = (cmd: string) => {
    const command = cmd.trim().toLowerCase();
    const newLine: TerminalLine = {
      id: Date.now().toString(),
      type: 'input',
      content: `$ ${cmd}`,
      timestamp: new Date(),
    };

    setLines((prev) => [...prev, newLine]);
    setHistory((prev) => [...prev, cmd]);
    setHistoryIndex(-1);

    // Simulate command execution
    setTimeout(() => {
      let output = '';
      if (command === 'help') {
        output = `Available commands:
  help     - Show this help message
  status   - Show robot status
  move     - Move robot (usage: move x y z)
  stop     - Stop robot
  home     - Move robot to home position
  clear    - Clear terminal
  version  - Show version information`;
      } else if (command === 'status') {
        output = 'Robot Status: Connected\nPosition: (0.0, 0.0, 0.0)\nBattery: 100%';
      } else if (command.startsWith('move')) {
        const parts = command.split(' ');
        if (parts.length === 4) {
          output = `Moving robot to (${parts[1]}, ${parts[2]}, ${parts[3]})...`;
        } else {
          output = 'Error: Usage: move x y z';
        }
      } else if (command === 'stop') {
        output = 'Robot stopped';
      } else if (command === 'home') {
        output = 'Moving robot to home position...';
      } else if (command === 'clear') {
        setLines([]);
        return;
      } else if (command === 'version') {
        output = 'Robot Movement AI v1.0.0';
      } else {
        output = `Command not found: ${cmd}. Type "help" for available commands.`;
      }

      const outputLine: TerminalLine = {
        id: (Date.now() + 1).toString(),
        type: output.startsWith('Error') ? 'error' : 'output',
        content: output,
        timestamp: new Date(),
      };
      setLines((prev) => [...prev, outputLine]);
    }, 100);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      executeCommand(input);
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (history.length > 0) {
        const newIndex = historyIndex === -1 ? history.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setInput(history[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex !== -1) {
        const newIndex = historyIndex + 1;
        if (newIndex >= history.length) {
          setHistoryIndex(-1);
          setInput('');
        } else {
          setHistoryIndex(newIndex);
          setInput(history[newIndex]);
        }
      }
    }
  };

  const handleClear = () => {
    setLines([]);
    toast.info('Terminal limpiado');
  };

  const getLineColor = (type: string) => {
    switch (type) {
      case 'input':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-300';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <TerminalIcon className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Terminal</h3>
          </div>
          <button
            onClick={handleClear}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
          >
            <Trash2 className="w-4 h-4" />
            Limpiar
          </button>
        </div>

        {/* Terminal Output */}
        <div
          ref={terminalRef}
          className="mb-4 h-96 bg-gray-900 rounded-lg p-4 overflow-y-auto font-mono text-sm border border-gray-600"
        >
          {lines.map((line) => (
            <div key={line.id} className={`mb-1 ${getLineColor(line.type)}`}>
              {line.content.split('\n').map((part, i) => (
                <div key={i}>{part}</div>
              ))}
            </div>
          ))}
        </div>

        {/* Terminal Input */}
        <form onSubmit={handleSubmit}>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-green-400 font-mono text-sm">
                $
              </span>
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Escribe un comando..."
                className="w-full pl-8 pr-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <button
              type="submit"
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              Enviar
            </button>
          </div>
        </form>

        {/* Help */}
        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <p className="text-xs text-blue-400">
            Tip: Usa las flechas ↑↓ para navegar el historial de comandos
          </p>
        </div>
      </div>
    </div>
  );
}


