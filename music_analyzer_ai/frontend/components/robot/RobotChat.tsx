'use client';

import { Send, Bot, User, AlertCircle } from 'lucide-react';

interface RobotChatProps {
  messages: Array<{ role: 'user' | 'robot'; content: string }>;
  inputMessage: string;
  setInputMessage: (message: string) => void;
  onSendMessage: () => void;
  isLoading: boolean;
  isConnected: boolean;
}

export function RobotChat({
  messages,
  inputMessage,
  setInputMessage,
  onSendMessage,
  isLoading,
  isConnected,
}: RobotChatProps) {
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSendMessage();
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 flex flex-col h-[600px]">
      {/* Chat Header */}
      <div className="p-4 border-b border-white/20">
        <div className="flex items-center gap-3">
          <Bot className="w-6 h-6 text-green-300" />
          <h2 className="text-xl font-semibold text-white">Chat con Robot</h2>
          {!isConnected && (
            <div className="flex items-center gap-2 text-yellow-300 text-sm">
              <AlertCircle className="w-4 h-4" />
              <span>Desconectado</span>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">
            <Bot className="w-12 h-12 mx-auto mb-4 text-gray-500" />
            <p>Envía un mensaje para comenzar</p>
            <p className="text-sm mt-2">
              Ejemplo: "Muévete a la posición x=1, y=2, z=3"
            </p>
          </div>
        ) : (
          messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'robot' && (
                <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              <div
                className={`max-w-[70%] rounded-lg p-3 ${
                  message.role === 'user'
                    ? 'bg-green-600 text-white'
                    : 'bg-white/10 text-gray-100'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              </div>
              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex gap-3 justify-start">
            <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-75" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/20">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              isConnected
                ? 'Escribe un comando o pregunta...'
                : 'Conecta el robot primero...'
            }
            disabled={!isConnected || isLoading}
            className="flex-1 px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-green-400 disabled:opacity-50"
          />
          <button
            onClick={onSendMessage}
            disabled={!isConnected || isLoading || !inputMessage.trim()}
            className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

