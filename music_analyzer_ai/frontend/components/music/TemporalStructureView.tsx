'use client';

import { useState, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Clock, Loader2, Music } from 'lucide-react';
import toast from 'react-hot-toast';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface TemporalStructureViewProps {
  trackId: string;
}

export function TemporalStructureView({ trackId }: TemporalStructureViewProps) {
  const structureMutation = useMutation({
    mutationFn: () => musicApiService.getTemporalStructure(trackId),
    onSuccess: () => {
      toast.success('Análisis temporal completado');
    },
    onError: () => {
      toast.error('Error al obtener estructura temporal');
    },
  });

  useEffect(() => {
    if (trackId) {
      structureMutation.mutate();
    }
  }, [trackId]);

  if (structureMutation.isPending) {
    return (
      <div className="bg-white/5 rounded-lg p-8 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-purple-300 animate-spin" />
      </div>
    );
  }

  if (!structureMutation.data) {
    return null;
  }

  const sections = structureMutation.data.sections || [];
  const chartData = sections.map((section: any, idx: number) => ({
    time: section.start || idx * 30,
    duration: section.duration || 30,
    confidence: section.confidence || 0.8,
    type: section.type || 'verse',
  }));

  return (
    <div className="bg-white/5 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Estructura Temporal</h3>
      </div>

      {/* Sections Timeline */}
      <div className="mb-4">
        <div className="flex gap-1 overflow-x-auto pb-2">
          {sections.map((section: any, idx: number) => (
            <div
              key={idx}
              className="flex-shrink-0 px-3 py-2 bg-purple-500/30 rounded-lg text-center min-w-[80px]"
            >
              <p className="text-xs text-gray-400 mb-1">
                {Math.round(section.start || 0)}s
              </p>
              <p className="text-sm font-semibold text-white capitalize">
                {section.type || 'section'}
              </p>
              <p className="text-xs text-gray-400">
                {Math.round(section.duration || 0)}s
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Chart */}
      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#a855f7" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
            <XAxis
              dataKey="time"
              stroke="#e2e8f0"
              tick={{ fill: '#e2e8f0', fontSize: 10 }}
            />
            <YAxis
              stroke="#e2e8f0"
              tick={{ fill: '#e2e8f0', fontSize: 10 }}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a2e',
                border: '1px solid #4a5568',
                borderRadius: '8px',
                color: '#fff',
              }}
            />
            <Area
              type="monotone"
              dataKey="confidence"
              stroke="#a855f7"
              fillOpacity={1}
              fill="url(#colorConfidence)"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}

      {/* Section Details */}
      <div className="mt-4 space-y-2">
        {sections.slice(0, 5).map((section: any, idx: number) => (
          <div key={idx} className="flex items-center justify-between p-2 bg-white/5 rounded">
            <div className="flex items-center gap-2">
              <Music className="w-4 h-4 text-purple-300" />
              <span className="text-sm text-white capitalize">{section.type}</span>
            </div>
            <div className="text-xs text-gray-400">
              {Math.round(section.start || 0)}s - {Math.round((section.start || 0) + (section.duration || 0))}s
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

