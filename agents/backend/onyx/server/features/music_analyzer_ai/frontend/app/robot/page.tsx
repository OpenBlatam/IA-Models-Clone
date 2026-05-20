'use client';

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { robotApiService, type ChatMessage, type Position } from '@/lib/api/robot-api';
import { Bot, Send, Power, PowerOff, AlertTriangle, Activity } from 'lucide-react';
import toast from 'react-hot-toast';
import { RobotChat } from '@/components/robot/RobotChat';
import { RobotStatus } from '@/components/robot/RobotStatus';
import { RobotControls } from '@/components/robot/RobotControls';

export default function RobotPage() {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'robot'; content: string }>>([]);
  const [inputMessage, setInputMessage] = useState('');
  const queryClient = useQueryClient();

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['robot-status'],
    queryFn: () => robotApiService.getStatus(),
    refetchInterval: 2000,
  });

  const { data: metrics } = useQuery({
    queryKey: ['robot-metrics'],
    queryFn: () => robotApiService.getMetrics(),
    enabled: status?.connected,
    refetchInterval: 5000,
  });

  const connectMutation = useMutation({
    mutationFn: () => robotApiService.connect(),
    onSuccess: () => {
      toast.success('Robot conectado');
      refetchStatus();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al conectar el robot');
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: () => robotApiService.disconnect(),
    onSuccess: () => {
      toast.success('Robot desconectado');
      refetchStatus();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al desconectar el robot');
    },
  });

  const chatMutation = useMutation({
    mutationFn: (message: string) => robotApiService.chat(message),
    onSuccess: (response) => {
      setMessages((prev) => [
        ...prev,
        { role: 'robot', content: response.message },
      ]);
      if (response.action) {
        toast.success(`Acción: ${response.action}`);
      }
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al enviar mensaje');
    },
  });

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;

    setMessages((prev) => [...prev, { role: 'user', content: inputMessage }]);
    chatMutation.mutate(inputMessage);
    setInputMessage('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-900 via-emerald-900 to-green-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bot className="w-10 h-10 text-green-300" />
              <div>
                <h1 className="text-4xl font-bold text-white">Robot Movement AI</h1>
                <p className="text-gray-300">
                  Controla robots mediante chat y comandos naturales
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {status?.connected ? (
                <button
                  onClick={() => disconnectMutation.mutate()}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                >
                  <PowerOff className="w-5 h-5" />
                  Desconectar
                </button>
              ) : (
                <button
                  onClick={() => connectMutation.mutate()}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  <Power className="w-5 h-5" />
                  Conectar
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column - Status & Controls */}
          <div className="space-y-6">
            <RobotStatus status={status} metrics={metrics} />
            {status?.connected && <RobotControls />}
          </div>

          {/* Right Column - Chat */}
          <div className="lg:col-span-2">
            <RobotChat
              messages={messages}
              inputMessage={inputMessage}
              setInputMessage={setInputMessage}
              onSendMessage={handleSendMessage}
              isLoading={chatMutation.isPending}
              isConnected={status?.connected || false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

