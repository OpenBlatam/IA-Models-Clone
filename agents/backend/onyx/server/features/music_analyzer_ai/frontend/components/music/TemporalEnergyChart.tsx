'use client';

import { useState, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface TemporalEnergyChartProps {
  trackId: string;
}

export function TemporalEnergyChart({ trackId }: TemporalEnergyChartProps) {
  const [energyData, setEnergyData] = useState<any>(null);

  const energyMutation = useMutation({
    mutationFn: () => musicApiService.getTemporalEnergy(trackId),
    onSuccess: (data) => {
      if (data.energy_progression) {
        const chartData = data.energy_progression.map((point: any, idx: number) => ({
          time: `${Math.round(point.time || idx * 10)}s`,
          energy: Math.round((point.energy || 0) * 100),
        }));
        setEnergyData(chartData);
      }
    },
    onError: (error: any) => {
      toast.error('Error al obtener datos temporales');
    },
  });

  useEffect(() => {
    if (trackId) {
      energyMutation.mutate();
    }
  }, [trackId]);

  if (energyMutation.isPending) {
    return (
      <div className="bg-white/5 rounded-lg p-8 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-purple-300 animate-spin" />
      </div>
    );
  }

  if (!energyData || energyData.length === 0) {
    return (
      <div className="bg-white/5 rounded-lg p-8 text-center">
        <p className="text-gray-400">No hay datos temporales disponibles</p>
      </div>
    );
  }

  return (
    <div className="bg-white/5 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Progresión de Energía</h3>
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <AreaChart data={energyData}>
          <defs>
            <linearGradient id="colorEnergy" x1="0" y1="0" x2="0" y2="1">
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
            domain={[0, 100]}
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
            dataKey="energy"
            stroke="#a855f7"
            fillOpacity={1}
            fill="url(#colorEnergy)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

