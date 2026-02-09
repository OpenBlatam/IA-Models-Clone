'use client';

import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { FiUsers, FiUser } from 'react-icons/fi';

interface User {
  id: string;
  name: string;
  color: string;
  cursor?: { line: number; column: number };
}

interface CollaborativeEditorProps {
  content: string;
  onChange: (content: string) => void;
  taskId: string;
}

export default function CollaborativeEditor({
  content,
  onChange,
  taskId,
}: CollaborativeEditorProps) {
  const [users, setUsers] = useState<User[]>([]);
  const [localUser] = useState<User>({
    id: `user-${Date.now()}`,
    name: 'Usuario',
    color: '#3B82F6',
  });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Simulate collaborative editing (in production, use WebSocket)
    const interval = setInterval(() => {
      // This would be replaced with real WebSocket connection
      // For now, just show local user
      setUsers([localUser]);
    }, 1000);

    return () => clearInterval(interval);
  }, [localUser]);

  const handleContentChange = (newContent: string) => {
    onChange(newContent);
    // In production, broadcast changes via WebSocket
  };

  return (
    <div className="relative">
      {/* Users indicator */}
      {users.length > 0 && (
        <div className="absolute top-2 right-2 flex items-center gap-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-1.5 shadow-sm z-10">
          <FiUsers size={16} className="text-gray-400" />
          <div className="flex items-center gap-1">
            {users.map((user) => (
              <div
                key={user.id}
                className="flex items-center gap-1.5"
                title={user.name}
              >
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-medium"
                  style={{ backgroundColor: user.color }}
                >
                  <FiUser size={12} />
                </div>
              </div>
            ))}
          </div>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {users.length} {users.length === 1 ? 'usuario' : 'usuarios'}
          </span>
        </div>
      )}

      <textarea
        ref={textareaRef}
        value={content}
        onChange={(e) => handleContentChange(e.target.value)}
        className="w-full p-4 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-800 text-gray-900 dark:text-white min-h-[300px] font-mono text-sm"
        placeholder="Escribe aquí... (Modo colaborativo)"
      />
    </div>
  );
}


