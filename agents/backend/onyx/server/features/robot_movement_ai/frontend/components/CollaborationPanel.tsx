'use client';

import { useState, useEffect } from 'react';
import { Users, UserPlus, MessageCircle, Video, Share2 } from 'lucide-react';
import { format } from 'date-fns';

interface Collaborator {
  id: string;
  name: string;
  role: 'admin' | 'operator' | 'viewer';
  status: 'online' | 'offline';
  lastSeen: string;
  currentAction?: string;
}

interface ChatMessage {
  id: string;
  user: string;
  message: string;
  timestamp: string;
}

export default function CollaborationPanel() {
  const [collaborators, setCollaborators] = useState<Collaborator[]>([
    {
      id: '1',
      name: 'Usuario Actual',
      role: 'admin',
      status: 'online',
      lastSeen: new Date().toISOString(),
      currentAction: 'Controlando robot',
    },
    {
      id: '2',
      name: 'Operador 1',
      role: 'operator',
      status: 'online',
      lastSeen: new Date().toISOString(),
      currentAction: 'Monitoreando',
    },
  ]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isSharing, setIsSharing] = useState(false);

  const handleSendMessage = () => {
    if (!newMessage.trim()) return;

    const message: ChatMessage = {
      id: Date.now().toString(),
      user: 'Usuario Actual',
      message: newMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages([...messages, message]);
    setNewMessage('');
  };

  const handleInvite = () => {
    // Simulate invite
    const newCollaborator: Collaborator = {
      id: Date.now().toString(),
      name: `Usuario ${collaborators.length + 1}`,
      role: 'viewer',
      status: 'online',
      lastSeen: new Date().toISOString(),
    };
    setCollaborators([...collaborators, newCollaborator]);
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Robot Movement AI',
          text: 'Únete a la sesión de control del robot',
          url: window.location.href,
        });
        setIsSharing(true);
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(window.location.href);
      setIsSharing(true);
      setTimeout(() => setIsSharing(false), 2000);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Collaborators List */}
      <div className="lg:col-span-1 space-y-4">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Users className="w-5 h-5 text-primary-400" />
              <h3 className="text-lg font-semibold text-white">Colaboradores</h3>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleInvite}
                className="p-2 bg-primary-600 hover:bg-primary-700 text-white rounded transition-colors"
                title="Invitar"
              >
                <UserPlus className="w-4 h-4" />
              </button>
              <button
                onClick={handleShare}
                className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                title="Compartir"
              >
                <Share2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="space-y-2">
            {collaborators.map((collab) => (
              <div
                key={collab.id}
                className="p-3 bg-gray-700/50 rounded-lg border border-gray-600"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        collab.status === 'online' ? 'bg-green-400' : 'bg-gray-400'
                      }`}
                    />
                    <span className="text-white font-medium">{collab.name}</span>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      collab.role === 'admin'
                        ? 'bg-red-500/20 text-red-400'
                        : collab.role === 'operator'
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'bg-gray-500/20 text-gray-400'
                    }`}
                  >
                    {collab.role}
                  </span>
                </div>
                {collab.currentAction && (
                  <p className="text-xs text-gray-400">{collab.currentAction}</p>
                )}
                <p className="text-xs text-gray-500 mt-1">
                  {format(new Date(collab.lastSeen), 'HH:mm')}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Video Call */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            <Video className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Llamada de Video</h3>
          </div>
          <button className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
            Iniciar Llamada
          </button>
        </div>
      </div>

      {/* Chat */}
      <div className="lg:col-span-2">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 flex flex-col h-[600px]">
          <div className="p-4 border-b border-gray-700 flex items-center gap-2">
            <MessageCircle className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Chat de Colaboración</h3>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-400 mt-8">
                <MessageCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No hay mensajes. Comienza la conversación.</p>
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className="flex flex-col">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-white">{msg.user}</span>
                    <span className="text-xs text-gray-400">
                      {format(new Date(msg.timestamp), 'HH:mm')}
                    </span>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <p className="text-gray-200">{msg.message}</p>
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="p-4 border-t border-gray-700">
            <div className="flex gap-2">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Escribe un mensaje..."
                className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <button
                onClick={handleSendMessage}
                className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
              >
                Enviar
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

