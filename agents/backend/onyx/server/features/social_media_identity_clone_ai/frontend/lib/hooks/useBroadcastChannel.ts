import { useEffect, useState, useCallback, useRef } from 'react';

interface UseBroadcastChannelOptions {
  onMessage?: (event: MessageEvent) => void;
}

export const useBroadcastChannel = (channelName: string, options: UseBroadcastChannelOptions = {}) => {
  const { onMessage } = options;
  const [isSupported, setIsSupported] = useState(false);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const channelRef = useRef<BroadcastChannel | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !('BroadcastChannel' in window)) {
      return;
    }

    setIsSupported(true);
    const channel = new BroadcastChannel(channelName);
    channelRef.current = channel;

    channel.onmessage = (event) => {
      setLastMessage(event);
      if (onMessage) {
        onMessage(event);
      }
    };

    return () => {
      channel.close();
    };
  }, [channelName, onMessage]);

  const postMessage = useCallback(
    (message: unknown) => {
      if (channelRef.current) {
        channelRef.current.postMessage(message);
      }
    },
    []
  );

  const close = useCallback(() => {
    if (channelRef.current) {
      channelRef.current.close();
      channelRef.current = null;
    }
  }, []);

  return {
    isSupported,
    lastMessage,
    postMessage,
    close,
  };
};



