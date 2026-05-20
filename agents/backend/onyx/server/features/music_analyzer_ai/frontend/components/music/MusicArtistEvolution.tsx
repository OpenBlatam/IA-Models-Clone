'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { User, TrendingUp, Music, Calendar } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MusicArtistEvolutionProps {
  artistName?: string;
}

export function MusicArtistEvolution({ artistName }: MusicArtistEvolutionProps) {
  const [selectedArtist, setSelectedArtist] = useState(artistName || '');

  const { data: evolution, isLoading } = useQuery({
    queryKey: ['artist-evolution', selectedArtist],
    queryFn: () => musicApiService.getArtistEvolution?.(selectedArtist) || Promise.resolve({ evolution: [] }),
    enabled: !!selectedArtist,
  });

  const evolutionData = evolution?.evolution || [];

  // Preparar datos para el gráfico
  const chartData = evolutionData.map((item: any) => ({
    year: item.year || 'N/A',
    popularity: item.popularity || 0,
    tracks: item.track_count || 0,
  }));

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <User className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Evolución del Artista</h3>
      </div>

      <div className="mb-4">
        <input
          type="text"
          value={selectedArtist}
          onChange={(e) => setSelectedArtist(e.target.value)}
          placeholder="Buscar artista..."
          className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-2 border-purple-300 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : evolutionData.length === 0 ? (
        <div className="text-center py-12">
          <User className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">Ingresa un artista para ver su evolución</p>
        </div>
      ) : (
        <div className="space-y-6">
          {chartData.length > 0 && (
            <div>
              <h4 className="text-white font-medium mb-3">Evolución de Popularidad</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                  <XAxis dataKey="year" stroke="#ffffff80" />
                  <YAxis stroke="#ffffff80" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(0, 0, 0, 0.8)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '8px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="popularity"
                    stroke="#a855f7"
                    strokeWidth={2}
                    dot={{ fill: '#a855f7' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="space-y-2">
            <h4 className="text-white font-medium mb-3">Historial por Año</h4>
            {evolutionData.map((item: any, idx: number) => (
              <div
                key={idx}
                className="flex items-center gap-3 p-3 bg-white/5 rounded-lg border border-white/10"
              >
                <div className="w-10 h-10 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                  <Calendar className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium">{item.year || 'Año desconocido'}</p>
                  <p className="text-sm text-gray-300">
                    {item.track_count || 0} canciones • Popularidad: {item.popularity || 0}%
                  </p>
                </div>
                {item.trend && (
                  <div className="flex items-center gap-1">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-green-400">
                      {item.trend > 0 ? '+' : ''}{item.trend}%
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


