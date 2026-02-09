/**
 * useChat - Hook Personalizado para Chat
 * =======================================
 * 
 * Hook para gestionar el estado y lógica del chat
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  error?: boolean;
  metadata?: Record<string, any>;
}

export interface UseChatOptions {
  initialMessages?: Message[];
  onSendMessage?: (message: string) => Promise<string>;
  maxMessages?: number;
  autoSave?: boolean;
  saveKey?: string;
}

export interface UseChatReturn {
  messages: Message[];
  input: string;
  isLoading: boolean;
  error: string | null;
  setInput: (value: string) => void;
  sendMessage: () => Promise<void>;
  clearMessages: () => void;
  retryLastMessage: () => Promise<void>;
  editMessage: (id: string, newContent: string) => void;
  deleteMessage: (id: string) => void;
  addSystemMessage: (content: string) => void;
}

export const useChat = (options: UseChatOptions = {}): UseChatReturn => {
  const {
    initialMessages = [],
    onSendMessage,
    maxMessages = 1000,
    autoSave = true,
    saveKey = 'chat-messages',
  } = options;

  const [messages, setMessages] = useState<Message[]>(() => {
    if (autoSave) {
      const saved = localStorage.getItem(saveKey);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          return parsed.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          }));
        } catch {
          return initialMessages;
        }
      }
    }
    return initialMessages;
  });

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Guardar mensajes automáticamente
  useEffect(() => {
    if (autoSave && messages.length > 0) {
      localStorage.setItem(saveKey, JSON.stringify(messages));
    }
  }, [messages, autoSave, saveKey]);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading || !onSendMessage) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => {
      const newMessages = [...prev, userMessage];
      return newMessages.slice(-maxMessages);
    });

    const currentInput = input;
    setInput('');
    setError(null);
    setIsLoading(true);

    // Mensaje de carga
    const loadingId = `loading-${Date.now()}`;
    const loadingMessage: Message = {
      id: loadingId,
      role: 'assistant',
      content: '...',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, loadingMessage]);

    try {
      // Crear nuevo AbortController
      abortControllerRef.current = new AbortController();
      
      const response = await onSendMessage(currentInput);

      setMessages((prev) => {
        const withoutLoading = prev.filter((msg) => msg.id !== loadingId);
        const assistantMessage: Message = {
          id: `msg-${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: response,
          timestamp: new Date(),
        };
        return [...withoutLoading, assistantMessage].slice(-maxMessages);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al enviar mensaje');

      setMessages((prev) => {
        const withoutLoading = prev.filter((msg) => msg.id !== loadingId);
        const errorMessage: Message = {
          id: `msg-${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: 'Lo siento, ocurrió un error al procesar tu mensaje.',
          timestamp: new Date(),
          error: true,
        };
        return [...withoutLoading, errorMessage].slice(-maxMessages);
      });
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [input, isLoading, onSendMessage, maxMessages]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    if (autoSave) {
      localStorage.removeItem(saveKey);
    }
  }, [autoSave, saveKey]);

  const retryLastMessage = useCallback(async () => {
    const lastUserMessage = [...messages].reverse().find((msg) => msg.role === 'user');
    if (!lastUserMessage || !onSendMessage) return;

    // Eliminar mensajes después del último mensaje del usuario
    const lastUserIndex = messages.findIndex((msg) => msg.id === lastUserMessage.id);
    setMessages((prev) => prev.slice(0, lastUserIndex + 1));

    setIsLoading(true);
    setError(null);

    try {
      const response = await onSendMessage(lastUserMessage.content);

      const assistantMessage: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage].slice(-maxMessages));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al reintentar');
    } finally {
      setIsLoading(false);
    }
  }, [messages, onSendMessage, maxMessages]);

  const editMessage = useCallback((id: string, newContent: string) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, content: newContent } : msg))
    );
  }, []);

  const deleteMessage = useCallback((id: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== id));
  }, []);

  const addSystemMessage = useCallback((content: string) => {
    const systemMessage: Message = {
      id: `system-${Date.now()}-${Math.random()}`,
      role: 'system',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, systemMessage].slice(-maxMessages));
  }, [maxMessages]);

  // Cleanup al desmontar
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    messages,
    input,
    isLoading,
    error,
    setInput,
    sendMessage,
    clearMessages,
    retryLastMessage,
    editMessage,
    deleteMessage,
    addSystemMessage,
  };
};


