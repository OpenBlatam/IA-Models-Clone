'use client';

import { useEffect, useRef } from 'react';
import { Music } from 'lucide-react';

interface MusicVisualizerProps {
  audioElement?: HTMLAudioElement | null;
  isPlaying: boolean;
}

export function MusicVisualizer({ audioElement, isPlaying }: MusicVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);

  useEffect(() => {
    if (!audioElement || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Crear AudioContext y Analyser
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaElementSource(audioElement);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    source.connect(analyser);
    analyser.connect(audioContext.destination);

    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    dataArrayRef.current = dataArray;

    const draw = () => {
      if (!isPlaying || !analyserRef.current || !dataArrayRef.current || !ctx) {
        animationFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      analyserRef.current.getByteFrequencyData(dataArrayRef.current);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArrayRef.current[i] / 255) * canvas.height;

        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, '#a855f7');
        gradient.addColorStop(1, '#ec4899');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }

      animationFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [audioElement, isPlaying]);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Music className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Visualizador</h3>
      </div>
      <canvas
        ref={canvasRef}
        width={600}
        height={200}
        className="w-full h-48 rounded-lg bg-gray-900"
      />
    </div>
  );
}


