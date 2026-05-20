'use client';

import { type Track } from '@/lib/api/music-api';
import { formatBPM, formatPercentage } from '@/lib/utils';

interface ComparisonMatrixProps {
  tracks: Track[];
  analysisData: any[];
}

export function ComparisonMatrix({ tracks, analysisData }: ComparisonMatrixProps) {
  if (tracks.length === 0 || analysisData.length === 0) return null;

  const features = ['energy', 'danceability', 'valence', 'tempo', 'key'];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 overflow-x-auto">
      <h3 className="text-lg font-semibold text-white mb-4">Matriz de Comparación</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/20">
            <th className="text-left p-2 text-gray-300">Característica</th>
            {tracks.map((track, idx) => (
              <th key={idx} className="text-left p-2 text-white max-w-[150px] truncate">
                {track.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature) => (
            <tr key={feature} className="border-b border-white/10">
              <td className="p-2 text-gray-300 capitalize">{feature}</td>
              {analysisData.map((analysis, idx) => {
                let value = 'N/A';
                if (feature === 'tempo' && analysis.musical_analysis?.tempo) {
                  value = formatBPM(analysis.musical_analysis.tempo.bpm);
                } else if (feature === 'key' && analysis.musical_analysis?.key_signature) {
                  value = analysis.musical_analysis.key_signature;
                } else if (analysis.technical_analysis?.[feature]) {
                  value = formatPercentage(analysis.technical_analysis[feature].value);
                }
                return (
                  <td key={idx} className="p-2 text-white">
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


