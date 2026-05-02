import { useEffect, useRef, useState, useCallback } from "react";
import * as Ably from "ably";

interface CursorPosition {
  x: number;
  y: number;
  userId: string;
}

interface UseAblyCursorOptions {
  ablyApiKey: string;
  channelName: string;
  userId: string;
}

export function useAblyCursor({ ablyApiKey, channelName, userId }: UseAblyCursorOptions) {
  const [remoteCursors, setRemoteCursors] = useState<Record<string, CursorPosition>>({});
  const ablyRef = useRef<Ably.Realtime | null>(null);
  const channelRef = useRef<Ably.RealtimeChannel | null>(null);

  // Inicializar Ably y canal
  useEffect(() => {
    if (!ablyApiKey || !channelName) return;
    ablyRef.current = new Ably.Realtime(ablyApiKey);
    channelRef.current = ablyRef.current.channels.get(channelName);

    // Escuchar posiciones de otros usuarios
    channelRef.current.subscribe("cursor", (msg) => {
      const { x, y, userId: senderId } = msg.data as CursorPosition;
      if (senderId !== userId) {
        setRemoteCursors((prev) => ({ ...prev, [senderId]: { x, y, userId: senderId } }));
      }
    });

    return () => {
      channelRef.current?.unsubscribe();
      ablyRef.current?.close();
    };
  }, [ablyApiKey, channelName, userId]);

  // Función para enviar la posición del mouse
  const sendCursor = useCallback((x: number, y: number) => {
    channelRef.current?.publish("cursor", { x, y, userId });
  }, [userId]);

  return {
    remoteCursors, // { [userId]: { x, y, userId } }
    sendCursor,
  };
}   