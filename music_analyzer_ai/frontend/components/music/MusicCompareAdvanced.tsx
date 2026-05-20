'use client';

import { useState } from 'react';
import { GitCompare, BarChart3, TrendingUp, TrendingDown } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicCompareAdvancedProps {
  tracks: Track[];
  analysisData?: any[];
}

export function MusicCompareAdvanced({ tracks, analysisData }: MusicCompareAdvancedProps) {
  const [selectedMetric, setSelectedMetric] = useState('energy');

  if (!analysisData || analysisData.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <GitCompare className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <p className="text-gray-400">Analiza las canciones primero para compararlas</p>
      </div>
    );
  }

  const metrics = ['energy', 'danceability', 'valence', 'tempo', 'popularity'];

  const getMetricValue = (analysis: any, metric: string) => {
    switch (metric) {
      case 'energy':
        return analysis.technical_analysis?.energy?.value || 0;
      case 'danceability':
        return analysis.technical_analysis?.danceability?.value || 0;
      case 'valence':
        return analysis.technical_analysis?.valence?.value || 0;
      case 'tempo':
        return analysis.musical_analysis?.tempo?.bpm || 0;
      case 'popularity':
        return analysis.track_basic_info?.popularity || 0;
      default:
        return 0;
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Comparación Avanzada</h3>
      </div>

      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Métrica</label>
        <select
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value)}
          className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
        >
          {metrics.map((metric) => (
            <option key={metric} value={metric}>
              {metric.charAt(0).toUpperCase() + metric.slice(1)}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-3">
        {tracks.map((track, idx) => {
          const analysis = analysisData[idx];
          const value = getMetricValue(analysis, selectedMetric);
          const maxValue = Math.max(...analysisData.map((a) => getMetricValue(a, selectedMetric)));

          return (
            <div key={track.id || idx} className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-white font-medium text-sm truncate flex-1">
                  {track.name}
                </span>
                <span className="text-sm text-gray-400 ml-2">
                  {selectedMetric === 'tempo' || selectedMetric === 'popularity'
                    ? Math.round(value)
                    : (value * 100).toFixed(0)}
                  {selectedMetric === 'tempo' ? ' BPM' : selectedMetric === 'popularity' ? '' : '%'}
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    value === maxValue ? 'bg-green-500' : 'bg-purple-500'
                  }`}
                  style={{
                    width: `${maxValue > 0 ? (value / maxValue) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


