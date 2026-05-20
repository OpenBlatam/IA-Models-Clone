import { useEffect, useRef, useState, useCallback } from 'react';

interface UseWebSocketOptions {
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
}

export const useWebSocket = (url: string | null, options: UseWebSocketOptions = {}) => {
  const {
    onOpen,
    onClose,
    onError,
    onMessage,
    reconnect = false,
    reconnectInterval = 3000,
    reconnectAttempts = 5,
  } = options;

  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    if (!url) {
      return;
    }

    try {
      const ws = new WebSocket(url);
      
      ws.onopen = (event) => {
        setIsConnected(true);
        reconnectAttemptsRef.current = 0;
        if (onOpen) {
          onOpen(event);
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        if (onClose) {
          onClose(event);
        }

        if (reconnect && reconnectAttemptsRef.current < reconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (event) => {
        if (onError) {
          onError(event);
        }
      };

      ws.onmessage = (event) => {
        setLastMessage(event);
        if (onMessage) {
          onMessage(event);
        }
      };

      setSocket(ws);
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }, [url, onOpen, onClose, onError, onMessage, reconnect, reconnectInterval, reconnectAttempts]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socket) {
        socket.close();
      }
    };
  }, [url]);

  const send = useCallback(
    (data: string | ArrayBuffer | Blob) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(data);
      }
    },
    [socket]
  );

  const close = useCallback(() => {
    if (socket) {
      socket.close();
    }
  }, [socket]);

  return {
    socket,
    isConnected,
    lastMessage,
    send,
    close,
    reconnect: connect,
  };
};



