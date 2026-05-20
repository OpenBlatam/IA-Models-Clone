'use client';

import { useCallback, useState } from 'react';

export function Conversation() {
  const [status, setStatus] = useState<'idle' | 'connected' | 'disconnected'>('idle');
  const [isSpeaking, setIsSpeaking] = useState(false);

  const startConversation = useCallback(async () => {
    try {
      // Request microphone permission
      await navigator.mediaDevices.getUserMedia({ audio: true });
      setStatus('connected');
      console.log('Connected to conversation');
    } catch (error) {
      console.error('Failed to start conversation:', error);
    }
  }, []);

  const stopConversation = useCallback(async () => {
    setStatus('disconnected');
    setIsSpeaking(false);
    console.log('Disconnected from conversation');
  }, []);

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex gap-2">
        <button
          onClick={startConversation}
          disabled={status === 'connected'}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
        >
          Start Conversation
        </button>
        <button
          onClick={stopConversation}
          disabled={status !== 'connected'}
          className="px-4 py-2 bg-red-500 text-white rounded disabled:bg-gray-300"
        >
          Stop Conversation
        </button>
      </div>

      <div className="flex flex-col items-center">
        <p>Status: {status}</p>
        <p>Agent is {isSpeaking ? 'speaking' : 'listening'}</p>
      </div>
    </div>
  );
}
