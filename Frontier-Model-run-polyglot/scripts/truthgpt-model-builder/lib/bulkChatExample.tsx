/**
 * Ejemplo de uso del hook useBulkChat
 * 
 * Este componente muestra cómo integrar la API de Bulk Chat
 * en tu aplicación React/Next.js
 */

'use client';

import { useState } from 'react';
import { useBulkChat } from './hooks';
import { Send, Pause, Play, Square, Loader2 } from 'lucide-react';

export function BulkChatExample() {
  const [input, setInput] = useState('');

  const {
    sessionId,
    messages,
    sendMessage,
    isConnected,
    isLoading,
    isPaused,
    error,
    createSession,
    pause,
    resume,
    stop,
    connectWebSocket,
    disconnectWebSocket,
  } = useBulkChat({
    apiUrl: process.env.NEXT_PUBLIC_BULK_CHAT_API_URL || 'http://localhost:8006',
    autoConnect: false, // Conectar manualmente
    autoContinue: true,
    enableWebSocket: true,
    onMessage: (message) => {
      console.log('Nuevo mensaje recibido:', message);
    },
    onError: (error) => {
      console.error('Error en Bulk Chat:', error);
    },
    onSessionCreated: (session) => {
      console.log('Sesión creada:', session);
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Si no hay sesión, crear una
    if (!sessionId) {
      await createSession(input);
      setInput('');
      return;
    }

    // Enviar mensaje
    await sendMessage(input);
    setInput('');
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 shadow-xl">
        {/* Header */}
        <div className="p-4 border-b border-slate-700 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Bulk Chat</h2>
            <div className="flex items-center gap-2 mt-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-sm text-slate-400">
                {isConnected ? 'Conectado' : 'Desconectado'}
                {sessionId && ` • Sesión: ${sessionId.slice(0, 8)}...`}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!sessionId ? (
              <button
                onClick={() => createSession()}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                Crear Sesión
              </button>
            ) : (
              <>
                {isPaused ? (
                  <button
                    onClick={resume}
                    disabled={isLoading}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                  >
                    <Play className="w-4 h-4" />
                    Reanudar
                  </button>
                ) : (
                  <button
                    onClick={pause}
                    disabled={isLoading}
                    className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                  >
                    <Pause className="w-4 h-4" />
                    Pausar
                  </button>
                )}
                <button
                  onClick={stop}
                  disabled={isLoading}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  <Square className="w-4 h-4" />
                  Detener
                </button>
                {!isConnected && (
                  <button
                    onClick={connectWebSocket}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                  >
                    Conectar WS
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="h-[500px] overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-slate-400 mt-10">
              <p className="text-lg mb-2">¡Hola! 👋</p>
              <p>Comienza una conversación o crea una sesión para empezar.</p>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`p-4 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600/20 ml-8'
                    : 'bg-slate-700/50 mr-8'
                }`}
              >
                <div className="flex items-start gap-2">
                  <div className="font-semibold text-slate-300">
                    {message.role === 'user' ? 'Tú' : 'Asistente'}:
                  </div>
                  <div className="flex-1 text-slate-200 whitespace-pre-wrap">
                    {message.content}
                  </div>
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex items-center gap-2 text-slate-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Procesando...</span>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-600/20 border border-red-600 rounded-lg">
              <p className="text-red-300">Error: {error.message}</p>
            </div>
          )}
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                sessionId
                  ? 'Escribe un mensaje...'
                  : 'Escribe un mensaje para crear la sesión...'
              }
              disabled={isLoading}
              className="flex-1 bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              Enviar
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}











