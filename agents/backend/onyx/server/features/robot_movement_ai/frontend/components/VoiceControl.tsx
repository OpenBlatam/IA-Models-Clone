'use client';

import { useState, useEffect } from 'react';
import { Mic, MicOff, Volume2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function VoiceControl() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isSupported, setIsSupported] = useState(false);

  useEffect(() => {
    setIsSupported('webkitSpeechRecognition' in window || 'SpeechRecognition' in window);
  }, []);

  const handleStartListening = () => {
    if (!isSupported) {
      toast.error('Reconocimiento de voz no soportado en este navegador');
      return;
    }

    setIsListening(true);
    setTranscript('');
    toast.info('Escuchando... Di tu comando');

    // Simulate voice recognition
    setTimeout(() => {
      setTranscript('Mover robot a posición x 0.5 y 0.3 z 0.2');
      setIsListening(false);
      toast.success('Comando reconocido');
    }, 3000);
  };

  const handleStopListening = () => {
    setIsListening(false);
    toast.info('Escuchado detenido');
  };

  const handleExecuteCommand = () => {
    if (transcript.trim()) {
      toast.success(`Ejecutando: ${transcript}`);
      setTranscript('');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Mic className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Control por Voz</h3>
        </div>

        {!isSupported && (
          <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/50 rounded-lg">
            <p className="text-sm text-yellow-400">
              El reconocimiento de voz no está disponible en este navegador. Usa Chrome o Edge para esta funcionalidad.
            </p>
          </div>
        )}

        {/* Voice Control */}
        <div className="mb-6">
          <div className="flex items-center justify-center mb-4">
            <button
              onClick={isListening ? handleStopListening : handleStartListening}
              disabled={!isSupported}
              className={`w-24 h-24 rounded-full flex items-center justify-center transition-all ${
                isListening
                  ? 'bg-red-600 hover:bg-red-700 animate-pulse'
                  : 'bg-primary-600 hover:bg-primary-700'
              } text-white disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isListening ? (
                <MicOff className="w-10 h-10" />
              ) : (
                <Mic className="w-10 h-10" />
              )}
            </button>
          </div>
          {isListening && (
            <div className="text-center mb-4">
              <div className="flex items-center justify-center gap-2 text-red-400">
                <Volume2 className="w-5 h-5 animate-pulse" />
                <span className="font-semibold">Escuchando...</span>
              </div>
            </div>
          )}
        </div>

        {/* Transcript */}
        {transcript && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-gray-300 mb-2">Comando Reconocido:</h4>
            <p className="text-white font-mono text-sm">{transcript}</p>
            <button
              onClick={handleExecuteCommand}
              className="mt-3 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
            >
              Ejecutar Comando
            </button>
          </div>
        )}

        {/* Commands Help */}
        <div className="p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">Comandos de Voz Disponibles:</h4>
          <ul className="text-xs text-gray-300 space-y-1">
            <li>• "Mover robot a posición X Y Z"</li>
            <li>• "Detener robot"</li>
            <li>• "Ir a home"</li>
            <li>• "Mostrar métricas"</li>
            <li>• "Abrir chat"</li>
          </ul>
        </div>
      </div>
    </div>
  );
}


