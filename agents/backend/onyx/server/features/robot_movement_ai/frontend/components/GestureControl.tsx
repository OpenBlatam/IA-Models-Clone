'use client';

import { useState, useRef, useEffect } from 'react';
import { Hand, Video } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function GestureControl() {
  const [isActive, setIsActive] = useState(false);
  const [gesture, setGesture] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsActive(true);
      toast.success('Control por gestos activado');
    } catch (error) {
      toast.error('Error al acceder a la cámara');
    }
  };

  const handleStop = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    setGesture(null);
    toast.info('Control por gestos desactivado');
  };

  useEffect(() => {
    if (isActive) {
      // Simulate gesture detection
      const interval = setInterval(() => {
        const gestures = ['fist', 'open', 'point', 'wave'];
        const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
        setGesture(randomGesture);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isActive]);

  const handleGestureAction = (gestureType: string) => {
    const actions: Record<string, string> = {
      fist: 'Detener robot',
      open: 'Mover adelante',
      point: 'Mover a posición',
      wave: 'Ir a home',
    };
    toast.info(`Ejecutando: ${actions[gestureType] || 'Acción desconocida'}`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Hand className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Control por Gestos</h3>
        </div>

        {/* Video Feed */}
        <div className="mb-6">
          <div className="relative bg-gray-900 rounded-lg overflow-hidden border border-gray-600">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-64 object-cover"
            />
            {!isActive && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                <div className="text-center">
                  <Video className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-400">Cámara inactiva</p>
                </div>
              </div>
            )}
            {gesture && isActive && (
              <div className="absolute top-4 left-4 px-3 py-1 bg-primary-600 rounded-lg">
                <p className="text-white font-semibold capitalize">{gesture}</p>
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="mb-6 flex gap-2">
          {!isActive ? (
            <button
              onClick={handleStart}
              className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Video className="w-4 h-4" />
              Activar Control por Gestos
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Video className="w-4 h-4" />
              Desactivar
            </button>
          )}
        </div>

        {/* Gesture Actions */}
        {gesture && (
          <div className="mb-6">
            <button
              onClick={() => handleGestureAction(gesture)}
              className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors font-semibold"
            >
              Ejecutar Acción: {gesture}
            </button>
          </div>
        )}

        {/* Gesture Guide */}
        <div className="p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">Gestos Disponibles:</h4>
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-300">
            <div>
              <span className="font-semibold">Puño:</span> Detener robot
            </div>
            <div>
              <span className="font-semibold">Mano abierta:</span> Mover adelante
            </div>
            <div>
              <span className="font-semibold">Señalar:</span> Mover a posición
            </div>
            <div>
              <span className="font-semibold">Saludar:</span> Ir a home
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


