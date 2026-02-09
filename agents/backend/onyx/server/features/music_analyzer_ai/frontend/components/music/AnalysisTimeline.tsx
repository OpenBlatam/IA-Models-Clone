'use client';

import { Clock, Music, TrendingUp } from 'lucide-react';

interface AnalysisTimelineProps {
  sections: Array<{
    type: string;
    start: number;
    duration: number;
    confidence?: number;
  }>;
}

export function AnalysisTimeline({ sections }: AnalysisTimelineProps) {
  const totalDuration = sections.reduce((sum, s) => sum + (s.duration || 0), 0);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Timeline de Estructura</h3>
      </div>

      <div className="relative">
        {/* Timeline Bar */}
        <div className="relative h-12 bg-gray-700 rounded-lg overflow-hidden mb-4">
          {sections.map((section, idx) => {
            const width = ((section.duration || 0) / totalDuration) * 100;
            const left = sections.slice(0, idx).reduce((sum, s) => sum + ((s.duration || 0) / totalDuration) * 100, 0);
            
            const colors: Record<string, string> = {
              intro: 'bg-blue-500',
              verse: 'bg-green-500',
              chorus: 'bg-purple-500',
              bridge: 'bg-yellow-500',
              outro: 'bg-red-500',
            };

            return (
              <div
                key={idx}
                className={`absolute h-full ${colors[section.type] || 'bg-gray-500'}`}
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                }}
                title={`${section.type}: ${Math.round(section.start)}s - ${Math.round(section.start + (section.duration || 0))}s`}
              />
            );
          })}
        </div>

        {/* Section Labels */}
        <div className="flex flex-wrap gap-2">
          {sections.map((section, idx) => (
            <div
              key={idx}
              className="flex items-center gap-2 px-3 py-1 bg-white/5 rounded-lg"
            >
              <div className={`w-3 h-3 rounded ${
                section.type === 'intro' ? 'bg-blue-500' :
                section.type === 'verse' ? 'bg-green-500' :
                section.type === 'chorus' ? 'bg-purple-500' :
                section.type === 'bridge' ? 'bg-yellow-500' :
                'bg-red-500'
              }`} />
              <span className="text-sm text-white capitalize">{section.type}</span>
              <span className="text-xs text-gray-400">
                {Math.round(section.start)}s
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


