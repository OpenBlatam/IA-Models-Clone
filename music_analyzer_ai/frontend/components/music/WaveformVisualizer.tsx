'use client';

import { useState, useEffect, useRef } from 'react';
import { type Track } from '@/lib/api/music-api';

interface WaveformVisualizerProps {
  track: Track;
  audioUrl?: string;
}

export function WaveformVisualizer({ track, audioUrl }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!canvasRef.current || !audioUrl) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Simular waveform (en producción, usar análisis de audio real)
    const drawWaveform = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 2;

      const centerY = height / 2;
      const bars = 100;
      const barWidth = width / bars;

      for (let i = 0; i < bars; i++) {
        const barHeight = Math.random() * (height / 2);
        const x = i * barWidth;
        ctx.beginPath();
        ctx.moveTo(x, centerY - barHeight);
        ctx.lineTo(x, centerY + barHeight);
        ctx.stroke();
      }
    };

    drawWaveform();
  }, [audioUrl]);

  if (!audioUrl) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <p className="text-gray-400 text-sm">Waveform no disponible</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <h3 className="text-sm font-semibold text-white mb-3">Waveform</h3>
      <canvas
        ref={canvasRef}
        width={600}
        height={100}
        className="w-full h-24 rounded-lg bg-gray-800"
      />
    </div>
  );
}


