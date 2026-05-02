"use client";

import { useEffect, useRef, useState } from 'react';
import { Room, RoomEvent, RemoteParticipant, RemoteTrackPublication, RemoteAudioTrack, LocalAudioTrack, createLocalAudioTrack, LocalTrack } from 'livekit-client';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

export default function MarketingAgentPage() {
  const [room, setRoom] = useState<Room | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const connectToRoom = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const newRoom = new Room();
        const response = await fetch('/api/livekit-token');
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Error al obtener el token');
        }
        const { token } = await response.json();

        const livekitUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL;
        if (!livekitUrl) {
          throw new Error('LiveKit URL no configurada');
        }

        await newRoom.connect(livekitUrl, token);

        newRoom.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
          if (
            participant.identity === 'marketing-bot' &&
            track.kind === 'audio' &&
            track instanceof RemoteAudioTrack
          ) {
            track.attach(audioRef.current!);
            toast.success('Respuesta del bot recibida');
          }
        });

        setRoom(newRoom);
        setIsConnected(true);
        toast.success('Conectado al asistente de marketing');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Error desconocido';
        setError(errorMessage);
        toast.error(`Error al conectar con el asistente: ${errorMessage}`);
      } finally {
        setIsLoading(false);
      }
    };

    connectToRoom();

    return () => {
      if (room) {
        room.disconnect();
      }
    };
  }, []);

  const toggleMute = async () => {
    if (!room) return;

    try {
      if (isMuted) {
        const audioTrack = await createLocalAudioTrack();
        await room.localParticipant.publishTrack(audioTrack);
      } else {
        room.localParticipant.getTrackPublications().forEach(pub => {
          if (pub.kind === 'audio') {
            room.localParticipant.unpublishTrack(pub.track! as LocalTrack);
          }
        });
      }
      setIsMuted(!isMuted);
    } catch (error) {
      toast.error('Error al cambiar el estado del micrófono');
    }
  };

  return (
    <div className="fixed top-4 right-4 p-6 bg-gray-900 text-white rounded-lg shadow-lg z-50 w-64 border-2 border-blue-500">
      <audio ref={audioRef} autoPlay />
      <div className="mb-4 text-base font-medium">
        {isLoading
          ? 'Conectando...'
          : error
            ? `Error: ${error}`
            : isConnected
              ? 'Conectado al asistente de marketing'
              : 'Error de conexión'}
      </div>
      <Button onClick={toggleMute} disabled={!isConnected || isLoading} className="w-full bg-blue-500 hover:bg-blue-600 text-base">
        {isMuted ? 'Activar Micrófono' : 'Desactivar Micrófono'}
      </Button>
    </div>
  );
}   