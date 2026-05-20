'use client';

import { TechnicalAnalysis } from '@/lib/api/music-api';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

interface AudioFeaturesChartProps {
  technicalAnalysis: TechnicalAnalysis;
}

export function AudioFeaturesChart({ technicalAnalysis }: AudioFeaturesChartProps) {
  const data = [
    {
      feature: 'Energía',
      value: technicalAnalysis.energy.value * 100,
      fullMark: 100,
    },
    {
      feature: 'Bailabilidad',
      value: technicalAnalysis.danceability.value * 100,
      fullMark: 100,
    },
    {
      feature: 'Valencia',
      value: technicalAnalysis.valence.value * 100,
      fullMark: 100,
    },
    {
      feature: 'Acústica',
      value: technicalAnalysis.acousticness.value * 100,
      fullMark: 100,
    },
    {
      feature: 'Instrumental',
      value: technicalAnalysis.instrumentalness.value * 100,
      fullMark: 100,
    },
    {
      feature: 'En Vivo',
      value: technicalAnalysis.liveness.value * 100,
      fullMark: 100,
    },
  ];

  return (
    <div className="bg-white/5 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-white mb-4">Características de Audio</h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="feature" tick={{ fill: '#e2e8f0', fontSize: 12 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#e2e8f0', fontSize: 10 }} />
          <Radar
            name="Valores"
            dataKey="value"
            stroke="#a855f7"
            fill="#a855f7"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

