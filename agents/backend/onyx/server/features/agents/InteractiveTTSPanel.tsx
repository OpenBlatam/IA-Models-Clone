"use client";

// components/InteractiveTTSPanel.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Room, RoomEvent, RemoteParticipant } from 'livekit-client';

// Iconos SVG simples (puedes reemplazarlos con una librería como lucide-react)
const SpeakerIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
  </svg>
);

const LoaderIcon = () => (
  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const UsersIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
    <circle cx="9" cy="7" r="4"></circle>
    <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
    <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
  </svg>
);

const LiveIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
    <circle cx="12" cy="12" r="10"></circle>
    <circle cx="12" cy="12" r="4"></circle>
  </svg>
);

// Configuración de LiveKit (deberían estar en variables de entorno)
const LIVEKIT_URL = process.env.NEXT_PUBLIC_LIVEKIT_URL || 'ws://localhost:7880'; // Reemplaza con tu URL de LiveKit
const TEMP_TOKEN = process.env.NEXT_PUBLIC_TEMP_LIVEKIT_TOKEN || 'tu_token_temporal_aqui'; // REEMPLAZA ESTO

const InteractiveTTSPanel: React.FC = () => {
  const [text, setText] = useState<string>('');
  const [isLoadingTTS, setIsLoadingTTS] = useState<boolean>(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const [room, setRoom] = useState<Room | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [participants, setParticipants] = useState<RemoteParticipant[]>([]);

  useEffect(() => {
    if (!LIVEKIT_URL || !TEMP_TOKEN || TEMP_TOKEN === 'tu_token_temporal_aqui') {
      return;
    }

    const newRoom = new Room({
      adaptiveStream: true,
      dynacast: true,
    });

    newRoom
      .on(RoomEvent.ParticipantConnected, (participant: RemoteParticipant) => {
        setParticipants((prev) => [...prev.filter(p => p.sid !== participant.sid), participant]);
      })
      .on(RoomEvent.ParticipantDisconnected, (participant: RemoteParticipant) => {
        setParticipants((prev) => prev.filter(p => p.sid !== participant.sid));
      })
      .on(RoomEvent.Disconnected, () => {
        setIsConnected(false);
        setParticipants([]);
      });

    const connectToRoom = async () => {
      try {
        await newRoom.connect(LIVEKIT_URL!, TEMP_TOKEN!);
        setIsConnected(true);
        setParticipants(Array.from(newRoom.remoteParticipants.values()));
        setRoom(newRoom);
      } catch (error) {
      }
    };

    connectToRoom();

    return () => {
      newRoom.disconnect();
    };
  }, []);

  const handleSynthesizeAndPlay = async () => {
    if (!text.trim()) {
      setTtsError('Por favor, introduce algún texto.');
      return;
    }
    setIsLoadingTTS(true);
    setTtsError(null);

    try {
      const response = await fetch('/api/tts-elevenlabs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Error HTTP: ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.play();
        audioRef.current.onended = () => {
          URL.revokeObjectURL(url);
        };
      }
    } catch (err: any) {
      setTtsError(err.message || 'Fallo al sintetizar la voz.');
    } finally {
      setIsLoadingTTS(false);
    }
  };

  return (
    <div className="sticky top-20 float-right w-[400px] bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.12)] border border-gray-700 overflow-hidden transform hover:scale-[1.02] transition-transform duration-200">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center">
              <SpeakerIcon />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">Voz IA</h2>
              <p className="text-sm text-gray-400">Genera voz natural con IA</p>
            </div>
          </div>
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
            {isConnected ? 'Conectado' : 'Desconectado'}
          </div>
        </div>

        <div className="space-y-4">
          <textarea
            className="w-full h-32 bg-gray-800/50 border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Escribe el texto que quieres convertir a voz..."
            disabled={isLoadingTTS}
          />

          <button
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-500 text-white font-medium rounded-xl hover:bg-blue-600 transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed shadow-lg hover:shadow-blue-500/25"
            onClick={handleSynthesizeAndPlay}
            disabled={isLoadingTTS}
          >
            {isLoadingTTS ? (
              <>
                <LoaderIcon />
                <span>Generando audio...</span>
              </>
            ) : (
              <>
                <SpeakerIcon />
                <span>Generar y reproducir</span>
              </>
            )}
          </button>

          {ttsError && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
              <p className="text-sm text-red-400">{ttsError}</p>
            </div>
          )}
        </div>

        <div className="mt-6 pt-6 border-t border-gray-700">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>Participantes en sala</span>
            <span className="font-medium text-white">{room?.numParticipants || 0}</span>
          </div>
          <div className="mt-3 space-y-2">
            {participants.map(p => (
              <div key={p.sid} className="flex items-center gap-3 text-sm text-gray-300">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                {p.identity}
              </div>
            ))}
          </div>
        </div>
      </div>
      <audio ref={audioRef} className="hidden" />
    </div>
  );
};

export default InteractiveTTSPanel;
