/**
 * Hook personalizado para integrar la API de Bulk Chat
 * 
 * Uso:
 * ```tsx
 * const { 
 *   sessionId, 
 *   messages, 
 *   sendMessage, 
 *   isConnected, 
 *   isLoading,
 *   pause,
 *   resume,
 *   stop 
 * } = useBulkChat({
 *   apiUrl: 'http://localhost:8006',
 *   autoConnect: true,
 *   onMessage: (msg) => console.log('Nuevo mensaje:', msg)
 * });
 * ```
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from 'react-hot-toast';

export interface BulkChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface BulkChatSession {
  session_id: string;
  state: string;
  is_paused: boolean;
  message_count: number;
  auto_continue: boolean;
}

export interface UseBulkChatOptions {
  apiUrl?: string;
  autoConnect?: boolean;
  initialMessage?: string;
  autoContinue?: boolean;
  onMessage?: (message: BulkChatMessage) => void;
  onError?: (error: Error) => void;
  onSessionCreated?: (session: BulkChatSession) => void;
  enableWebSocket?: boolean;
  enableNotifications?: boolean;
  enableSounds?: boolean;
}

export interface UseBulkChatReturn {
  // Estado
  sessionId: string | null;
  session: BulkChatSession | null;
  messages: BulkChatMessage[];
  isConnected: boolean;
  isLoading: boolean;
  isPaused: boolean;
  error: Error | null;
  reconnectAttempts: number;
  lastActivity: Date | null;
  isTyping: boolean;
  messageCount: number;
  isOnline: boolean;
  connectionQuality: 'excellent' | 'good' | 'poor' | 'offline';
  
  // Métodos REST
  createSession: (initialMessage?: string) => Promise<void>;
  sendMessage: (message: string) => Promise<void>;
  getMessages: (limit?: number) => Promise<BulkChatMessage[]>;
  pause: (reason?: string) => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  refreshMessages: () => Promise<void>;
  
  // WebSocket
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;
  
  // Utilidades
  clearError: () => void;
  retryConnection: () => Promise<void>;
}

const DEFAULT_API_URL = 'http://localhost:8006';

export function useBulkChat(options: UseBulkChatOptions = {}): UseBulkChatReturn {
  const {
    apiUrl = DEFAULT_API_URL,
    autoConnect = false,
    initialMessage,
    autoContinue = true,
    onMessage,
    onError,
    onSessionCreated,
    enableWebSocket = true,
    enableNotifications = true,
    enableSounds = false,
  } = options;

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [session, setSession] = useState<BulkChatSession | null>(null);
  const [messages, setMessages] = useState<BulkChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastActivity, setLastActivity] = useState<Date | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [messageCount, setMessageCount] = useState(0);
  const [pendingMessages, setPendingMessages] = useState<BulkChatMessage[]>([]);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [connectionQuality, setConnectionQuality] = useState<'excellent' | 'good' | 'poor' | 'offline'>('excellent');

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messagePollingRef = useRef<NodeJS.Timeout | null>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  // Detectar estado online/offline
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setConnectionQuality('good');
      if (sessionId && !isConnected) {
        connectWebSocket();
      }
    };
    
    const handleOffline = () => {
      setIsOnline(false);
      setConnectionQuality('offline');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, isConnected]);

  // Monitorear calidad de conexión
  useEffect(() => {
    if (!isConnected || !sessionId) return;

    const checkConnection = setInterval(async () => {
      try {
        const start = Date.now();
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        
        const response = await fetch(`${apiUrl}/health`, { 
          method: 'GET',
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const latency = Date.now() - start;
        
        if (response.ok) {
          if (latency < 100) {
            setConnectionQuality('excellent');
          } else if (latency < 500) {
            setConnectionQuality('good');
          } else {
            setConnectionQuality('poor');
          }
        } else {
          setConnectionQuality('poor');
        }
      } catch (e) {
        setConnectionQuality('poor');
      }
    }, 10000); // Check cada 10 segundos

    return () => clearInterval(checkConnection);
  }, [isConnected, sessionId, apiUrl]);

  // Limpiar al desmontar
  useEffect(() => {
    return () => {
      disconnectWebSocket();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (messagePollingRef.current) {
        clearInterval(messagePollingRef.current);
      }
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  // Auto-connect si está habilitado
  useEffect(() => {
    if (autoConnect && !sessionId && !isLoading) {
      createSession(initialMessage);
    }
  }, [autoConnect]);

  // Polling de mensajes cuando hay sesión activa y no hay WebSocket
  useEffect(() => {
    if (sessionId && !wsRef.current && !messagePollingRef.current) {
      messagePollingRef.current = setInterval(() => {
        refreshMessages();
      }, 2000); // Poll cada 2 segundos
    }

    return () => {
      if (messagePollingRef.current) {
        clearInterval(messagePollingRef.current);
        messagePollingRef.current = null;
      }
    };
  }, [sessionId, wsRef.current]);

  /**
   * Crear una nueva sesión de chat
   */
  const createSession = useCallback(async (msg?: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/api/v1/chat/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initial_message: msg || initialMessage || 'Hola',
          auto_continue: autoContinue,
          response_interval: 2.0,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: BulkChatSession = await response.json();
      setSessionId(data.session_id);
      setSession(data);
      setIsPaused(data.is_paused);

      if (onSessionCreated) {
        onSessionCreated(data);
      }

      // Solicitar permisos de notificación si está habilitado
      if (enableNotifications && 'Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission().catch(() => {});
      }

      // Cargar mensajes iniciales
      await refreshMessages();

      // Conectar WebSocket si está habilitado
      if (enableWebSocket) {
        connectWebSocket();
      }

      toast.success('Sesión de chat creada', { icon: '✅' });
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      if (onError) {
        onError(error);
      }
      toast.error(`Error al crear sesión: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [apiUrl, initialMessage, autoContinue, onSessionCreated, onError, enableWebSocket]);

  /**
   * Enviar un mensaje
   */
  const sendMessage = useCallback(async (message: string) => {
    if (!sessionId) {
      toast.error('No hay sesión activa. Crea una sesión primero.');
      return;
    }

    if (!message.trim()) {
      toast.error('El mensaje no puede estar vacío');
      return;
    }

    setIsLoading(true);
    setError(null);
    setIsTyping(true);

    // Optimistic update: agregar mensaje pendiente
    const tempId = `pending-${Date.now()}`;
    const pendingMessage: BulkChatMessage = {
      id: tempId,
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    };
    setPendingMessages(prev => [...prev, pendingMessage]);
    setMessages(prev => [...prev, pendingMessage]);

    try {
      // Enviar por WebSocket si está conectado
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'message',
          content: message,
        }));
        setIsLoading(false);
        return;
      }

      // Fallback a REST
      const response = await fetch(
        `${apiUrl}/api/v1/chat/sessions/${sessionId}/messages`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      // Remover de pending después de confirmar
      setPendingMessages(prev => prev.filter(msg => msg.id !== tempId));
      setMessageCount(prev => prev + 1);

      // Esperar un poco y refrescar mensajes
      setTimeout(() => {
        refreshMessages();
      }, 500);
    } catch (err) {
      // Remover mensaje pendiente en caso de error
      setPendingMessages(prev => prev.filter(msg => msg.id !== tempId));
      setMessages(prev => prev.filter(msg => msg.id !== tempId));
      
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      if (onError) {
        onError(error);
      }
      toast.error(`Error al enviar mensaje: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, apiUrl, onError, refreshMessages]);

  /**
   * Obtener mensajes de la sesión
   */
  const getMessages = useCallback(async (limit: number = 50): Promise<BulkChatMessage[]> => {
    if (!sessionId) {
      return [];
    }

    try {
      const response = await fetch(
        `${apiUrl}/api/v1/chat/sessions/${sessionId}/messages?limit=${limit}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.messages || [];
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      if (onError) {
        onError(error);
      }
      return [];
    }
  }, [sessionId, apiUrl, onError]);

  /**
   * Refrescar mensajes
   */
  const refreshMessages = useCallback(async () => {
    if (!sessionId) return;

    const newMessages = await getMessages();
    setMessages(newMessages);
  }, [sessionId, getMessages]);

  /**
   * Pausar la sesión
   */
  const pause = useCallback(async (reason?: string) => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      // Enviar por WebSocket si está conectado
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'pause' }));
        setIsPaused(true);
        setIsLoading(false);
        return;
      }

      // Fallback a REST
      const url = `${apiUrl}/api/v1/chat/sessions/${sessionId}/pause${reason ? `?reason=${encodeURIComponent(reason)}` : ''}`;
      const response = await fetch(url, { method: 'POST' });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setIsPaused(true);
      await refreshMessages();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      if (onError) {
        onError(error);
      }
      toast.error(`Error al pausar: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, apiUrl, refreshMessages, onError]);

  /**
   * Reanudar la sesión
   */
  const resume = useCallback(async () => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      // Enviar por WebSocket si está conectado
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'resume' }));
        setIsPaused(false);
        setIsLoading(false);
        return;
      }

      // Fallback a REST
      const response = await fetch(
        `${apiUrl}/api/v1/chat/sessions/${sessionId}/resume`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setIsPaused(false);
      await refreshMessages();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      if (onError) {
        onError(error);
      }
      toast.error(`Error al reanudar: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, apiUrl, refreshMessages, onError]);

  /**
   * Detener la sesión
   */
  const stop = useCallback(async () => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      // Enviar por WebSocket si está conectado
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
        disconnectWebSocket();
      }

      // Fallback a REST
      const response = await fetch(
        `${apiUrl}/api/v1/chat/sessions/${sessionId}/stop`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setSessionId(null);
      setSession(null);
      setMessages([]);
      setIsPaused(false);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      if (onError) {
        onError(error);
      }
      toast.error(`Error al detener: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, apiUrl, onError]);

  /**
   * Conectar WebSocket
   */
  const connectWebSocket = useCallback(() => {
    if (!sessionId) {
      toast.error('No hay sesión activa');
      return;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return; // Ya está conectado
    }

    try {
      const wsUrl = apiUrl.replace(/^http/, 'ws');
      const ws = new WebSocket(`${wsUrl}/ws/chat/${sessionId}`);

      ws.onopen = () => {
        setIsConnected(true);
        setReconnectAttempts(0); // Reset reconnect attempts on success
        setLastActivity(new Date());
        console.log('WebSocket conectado');
        
        // Limpiar polling si existe
        if (messagePollingRef.current) {
          clearInterval(messagePollingRef.current);
          messagePollingRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          setLastActivity(new Date()); // Update last activity
          const data = JSON.parse(event.data);

          switch (data.type) {
            case 'message':
              if (data.role === 'assistant') {
                setIsTyping(false);
                const newMessage: BulkChatMessage = {
                  id: data.id || Date.now().toString(),
                  role: 'assistant',
                  content: data.content,
                  timestamp: data.timestamp || new Date().toISOString(),
                };
                setMessages(prev => {
                  // Evitar duplicados
                  const exists = prev.some(msg => msg.id === newMessage.id);
                  if (exists) return prev;
                  setMessageCount(prev => prev + 1);
                  return [...prev, newMessage];
                });
                // Remover de pending si existe
                setPendingMessages(prev => prev.filter(msg => msg.id !== newMessage.id));
                
                // Notificación del navegador
                if (enableNotifications && 'Notification' in window && Notification.permission === 'granted') {
                  new Notification('Nuevo mensaje de Bulk Chat', {
                    body: newMessage.content.substring(0, 100) + (newMessage.content.length > 100 ? '...' : ''),
                    icon: '/favicon.ico',
                    tag: 'bulk-chat-message',
                  });
                }
                
                // Sonido opcional
                if (enableSounds) {
                  try {
                    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBzGL0fPTgjMGHm7A7+OaUxEMT6Tj8LdjHAY4kdfyzHksBSR3x/DdkEAKFF606euoVRQKRp/g8r5sIQcxi9Hz04IzBh5uwO/jmlMRDE+k4/C3YxwGOJHX8sx5LAUkd8fw3ZBAC');
                    audio.volume = 0.3;
                    audio.play().catch(() => {});
                  } catch (e) {
                    // Ignorar errores de audio
                  }
                }
                
                if (onMessage) {
                  onMessage(newMessage);
                }
              }
              break;

            case 'chunk':
              setIsTyping(true);
              // Actualizar último mensaje con chunk
              setMessages(prev => {
                const updated = [...prev];
                const lastMsg = updated[updated.length - 1];
                if (lastMsg && lastMsg.role === 'assistant') {
                  lastMsg.content += data.data;
                } else {
                  updated.push({
                    id: Date.now().toString(),
                    role: 'assistant',
                    content: data.data,
                    timestamp: new Date().toISOString(),
                  });
                }
                return updated;
              });
              // Reset typing indicator después de 2 segundos sin chunks
              if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
              }
              typingTimeoutRef.current = setTimeout(() => {
                setIsTyping(false);
              }, 2000);
              break;

            case 'session_state':
              if (data.data) {
                setSession(data.data);
                setIsPaused(data.data.is_paused || false);
              }
              break;

            case 'paused':
              setIsPaused(true);
              break;

            case 'resumed':
              setIsPaused(false);
              break;

            case 'stopped':
              setIsPaused(true);
              break;

            case 'error':
              const error = new Error(data.message || 'Error desconocido');
              setError(error);
              if (onError) {
                onError(error);
              }
              toast.error(`Error: ${data.message}`);
              break;
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        toast.error('Error en la conexión WebSocket');
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket desconectado');
        
        // Intentar reconectar con backoff exponencial
        if (sessionId && reconnectAttempts < maxReconnectAttempts) {
          const delay = reconnectDelay * Math.pow(2, reconnectAttempts);
          reconnectTimeoutRef.current = setTimeout(() => {
            if (sessionId) {
              setReconnectAttempts(prev => prev + 1);
              connectWebSocket();
            }
          }, delay);
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          console.warn('Max reconnection attempts reached, switching to polling');
          // Cambiar a polling si WebSocket falla
        }
      };

      wsRef.current = ws;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Error desconocido');
      setError(error);
      toast.error(`Error al conectar WebSocket: ${error.message}`);
    }
  }, [sessionId, apiUrl, onMessage, onError, enableNotifications, enableSounds, reconnectAttempts, maxReconnectAttempts, reconnectDelay]);

  /**
   * Desconectar WebSocket
   */
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  /**
   * Limpiar error
   */
  const clearError = useCallback(() => {
    setError(null);
    setReconnectAttempts(0);
  }, []);

  /**
   * Reintentar conexión
   */
  const retryConnection = useCallback(async () => {
    if (sessionId) {
      disconnectWebSocket();
      setReconnectAttempts(0);
      await new Promise(resolve => setTimeout(resolve, 1000));
      connectWebSocket();
    } else {
      await createSession();
    }
  }, [sessionId, disconnectWebSocket, connectWebSocket, createSession]);

  return {
    // Estado
    sessionId,
    session,
    messages,
    isConnected,
    isLoading,
    isPaused,
    error,

    // Métodos REST
    createSession,
    sendMessage,
    getMessages,
    pause,
    resume,
    stop,
    refreshMessages,

    // WebSocket
    connectWebSocket,
    disconnectWebSocket,

    // Utilidades
    clearError,
    retryConnection,
    reconnectAttempts,
    lastActivity,
    isTyping,
    messageCount,
    isOnline,
    connectionQuality,
  };
}

