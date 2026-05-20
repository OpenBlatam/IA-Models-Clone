import { useEffect, useRef, useState } from 'react';
import { Room, RoomEvent, RemoteAudioTrack, createLocalAudioTrack, LocalTrack } from 'livekit-client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Bot, Mic, MicOff } from 'lucide-react';

export function MarketingAgentCard() {
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
        toast.success('Conectado al agente de marketing');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Error desconocido';
        setError(errorMessage);
        toast.error(`Error al conectar con el agente: ${errorMessage}`);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    <Card className="shadow-lg border-blue-200">
      <CardHeader className="flex flex-row items-center gap-3 pb-2">
        <Bot className="h-7 w-7 text-blue-600" />
        <CardTitle className="text-lg font-bold">Agente de Marketing</CardTitle>
      </CardHeader>
      <CardContent>
        <audio ref={audioRef} autoPlay />
        <div className="mb-2 text-sm font-medium">
          {isLoading
            ? 'Conectando...'
            : error
              ? `Error: ${error}`
              : isConnected
                ? '¡Habla con tu asistente de marketing en tiempo real!'
                : 'Error de conexión'}
        </div>
        <Button onClick={toggleMute} disabled={!isConnected || isLoading} className="w-full bg-blue-500 hover:bg-blue-600 text-base flex items-center gap-2">
          {isMuted ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
          {isMuted ? 'Activar Micrófono' : 'Desactivar Micrófono'}
        </Button>
      </CardContent>
    </Card>
  );
} 