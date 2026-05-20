'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Clock, TrendingUp, TrendingDown } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MusicTemporalProps {
  trackId: string;
}

export function MusicTemporal({ trackId }: MusicTemporalProps) {
  const { data: temporal, isLoading: isLoadingTemporal } = useQuery({
    queryKey: ['temporal-structure', trackId],
    queryFn: () => musicApiService.getTemporalStructure?.(trackId) || Promise.resolve({}),
    enabled: !!trackId,
  });

  const { data: energy, isLoading: isLoadingEnergy } = useQuery({
    queryKey: ['temporal-energy', trackId],
    queryFn: () => musicApiService.getTemporalEnergy?.(trackId) || Promise.resolve({}),
    enabled: !!trackId,
  });

  const isLoading = isLoadingTemporal || isLoadingEnergy;

  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <Clock className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <p className="text-gray-400">Cargando análisis temporal...</p>
      </div>
    );
  }

  const energyData = energy?.energy_progression || [];
  const structureData = temporal?.sections || [];

  return (
    <div className="space-y-6">
      {energyData.length > 0 && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-purple-300" />
            <h3 className="text-lg font-semibold text-white">Progresión de Energía</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={energyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="time" stroke="#ffffff80" />
              <YAxis stroke="#ffffff80" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '8px',
                }}
              />
              <Area
                type="monotone"
                dataKey="energy"
                stroke="#a855f7"
                fill="#a855f7"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {structureData.length > 0 && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-5 h-5 text-purple-300" />
            <h3 className="text-lg font-semibold text-white">Estructura Temporal</h3>
          </div>
          <div className="space-y-2">
            {structureData.map((section: any, idx: number) => (
              <div
                key={idx}
                className="flex items-center gap-3 p-3 bg-white/5 rounded-lg border border-white/10"
              >
                <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-bold">{idx + 1}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium capitalize">
                    {section.type || 'Sección'}
                  </p>
                  <p className="text-sm text-gray-300">
                    {section.start_time || 0}s - {section.end_time || 0}s
                  </p>
                </div>
                {section.energy && (
                  <div className="flex items-center gap-2">
                    {section.energy > 0.5 ? (
                      <TrendingUp className="w-4 h-4 text-green-400" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-blue-400" />
                    )}
                    <span className="text-sm text-gray-400">
                      {Math.round(section.energy * 100)}%
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

