'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, Maximize2, Minimize2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function ARView() {
  const [isActive, setIsActive] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleStartAR = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsActive(true);
      toast.success('Vista AR activada');
    } catch (error) {
      toast.error('Error al acceder a la cámara');
    }
  };

  const handleStopAR = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    toast.info('Vista AR desactivada');
  };

  const handleToggleFullscreen = () => {
    if (!isFullscreen) {
      if (canvasRef.current?.requestFullscreen) {
        canvasRef.current.requestFullscreen();
      }
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const drawAR = () => {
      if (videoRef.current && canvasRef.current && isActive) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (ctx) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Draw AR overlay (robot position indicator)
          ctx.strokeStyle = '#0EA5E9';
          ctx.lineWidth = 3;
          ctx.strokeRect(canvas.width / 2 - 50, canvas.height / 2 - 50, 100, 100);
          ctx.fillStyle = 'rgba(14, 165, 233, 0.2)';
          ctx.fillRect(canvas.width / 2 - 50, canvas.height / 2 - 50, 100, 100);
        }
      }
      if (isActive) {
        requestAnimationFrame(drawAR);
      }
    };

    if (isActive) {
      drawAR();
    }
  }, [isActive]);

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Vista de Realidad Aumentada</h3>
          </div>
          {isActive && (
            <button
              onClick={handleToggleFullscreen}
              className="p-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          )}
        </div>

        {/* AR Canvas */}
        <div className="mb-6 relative bg-gray-900 rounded-lg overflow-hidden border border-gray-600">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-96 object-cover hidden"
          />
          <canvas
            ref={canvasRef}
            className="w-full h-96 object-cover"
          />
          {!isActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
              <div className="text-center">
                <Camera className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p className="text-gray-400 mb-4">Vista AR inactiva</p>
                <button
                  onClick={handleStartAR}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                >
                  Activar AR
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Controls */}
        {isActive && (
          <div className="flex gap-2">
            <button
              onClick={handleStopAR}
              className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
            >
              Detener AR
            </button>
          </div>
        )}

        {/* Info */}
        <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <p className="text-sm text-blue-400">
            La vista AR superpone información del robot sobre la cámara en tiempo real.
            Usa la cámara trasera para mejor experiencia.
          </p>
        </div>
      </div>
    </div>
  );
}


