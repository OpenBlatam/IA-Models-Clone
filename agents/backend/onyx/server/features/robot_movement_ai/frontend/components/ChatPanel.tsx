'use client';

import { useState } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { useScrollToBottom } from '@/lib/hooks/useScrollToBottom';
import { Send, Bot, User } from 'lucide-react';
import { format } from 'date-fns';

export default function ChatPanel() {
  const { chatMessages, sendChatMessage, isChatConnected } = useRobotStore();
  const [input, setInput] = useState('');
  const { containerRef: messagesEndRef, scrollToBottom } = useScrollToBottom<HTMLDivElement>(
    [chatMessages]
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const message = input.trim();
    setInput('');
    await sendChatMessage(message);
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm flex flex-col h-[600px]">
      {/* Header */}
      <div className="p-tesla-lg border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-xl font-semibold text-tesla-black">Chat con el Robot</h2>
        <div className="flex items-center gap-tesla-sm">
          <div
            className={`w-2 h-2 rounded-full ${
              isChatConnected ? 'bg-green-600' : 'bg-gray-400'
            }`}
          />
          <span className="text-sm text-tesla-gray-dark font-medium">
            {isChatConnected ? 'Conectado' : 'Desconectado'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div ref={messagesEndRef} className="flex-1 overflow-y-auto p-tesla-lg space-y-tesla-md">
        {chatMessages.length === 0 ? (
          <div className="text-center text-tesla-gray-dark mt-tesla-xl">
            <Bot className="w-12 h-12 mx-auto mb-tesla-md text-tesla-gray-light opacity-50" />
            <p className="font-medium">Envía un mensaje para comenzar</p>
            <p className="text-sm mt-tesla-sm text-tesla-gray-dark">
              Ejemplo: "move to (0.5, 0.3, 0.2)" o "go home"
            </p>
          </div>
        ) : (
          chatMessages.map((msg, index) => (
            <div
              key={index}
              className={`flex gap-tesla-sm ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-tesla-blue flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              <div
                className={`max-w-[70%] rounded-md p-tesla-sm ${
                  msg.role === 'user'
                    ? 'bg-tesla-blue text-white'
                    : 'bg-gray-100 text-tesla-black'
                }`}
              >
                <p className="text-sm">{msg.content}</p>
                <p className={`text-xs mt-tesla-xs ${msg.role === 'user' ? 'opacity-80' : 'text-tesla-gray-dark'}`}>
                  {format(msg.timestamp, 'HH:mm:ss')}
                </p>
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-tesla-gray-dark flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-tesla-lg border-t border-gray-200">
        <div className="flex gap-tesla-sm">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Escribe un comando o pregunta..."
            className="flex-1 px-4 py-3 bg-white border border-gray-300 rounded-md text-tesla-black placeholder-tesla-gray-light focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-tesla-lg py-tesla-sm bg-tesla-blue hover:bg-opacity-90 text-white font-medium rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-tesla-sm"
          >
            <Send className="w-5 h-5" />
            Enviar
          </button>
        </div>
      </form>
    </div>
  );
}

