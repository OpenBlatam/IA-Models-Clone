'use client';

import { useState, useEffect } from 'react';
import { Mic, MicOff } from 'lucide-react';
import toast from 'react-hot-toast';

interface VoiceCommandsProps {
  onCommand: (command: string) => void;
}

export function VoiceCommands({ onCommand }: VoiceCommandsProps) {
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState<any>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = false;
      recognitionInstance.lang = 'es-ES';

      recognitionInstance.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        onCommand(transcript);
        setIsListening(false);
        toast.success(`Comando: ${transcript}`);
      };

      recognitionInstance.onerror = () => {
        setIsListening(false);
        toast.error('Error en reconocimiento de voz');
      };

      setRecognition(recognitionInstance);
    }
  }, [onCommand]);

  const toggleListening = () => {
    if (!recognition) {
      toast.error('Reconocimiento de voz no disponible');
      return;
    }

    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      recognition.start();
      setIsListening(true);
      toast.info('Escuchando...');
    }
  };

  if (!recognition) {
    return null;
  }

  return (
    <button
      onClick={toggleListening}
      className={`p-3 rounded-full transition-colors ${
        isListening
          ? 'bg-red-600 hover:bg-red-700 animate-pulse'
          : 'bg-white/10 hover:bg-white/20'
      } text-white`}
      title="Comandos de voz"
    >
      {isListening ? (
        <MicOff className="w-5 h-5" />
      ) : (
        <Mic className="w-5 h-5" />
      )}
    </button>
  );
}


