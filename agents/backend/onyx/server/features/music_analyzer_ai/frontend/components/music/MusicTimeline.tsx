'use client';

import { Clock, Music } from 'lucide-react';

interface MusicTimelineProps {
  sections: Array<{
    start: number;
    end: number;
    type: string;
    name?: string;
  }>;
  duration: number;
}

export function MusicTimeline({ sections, duration }: MusicTimelineProps) {
  if (!sections || sections.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <Clock className="w-12 h-12 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400">No hay información de estructura temporal</p>
      </div>
    );
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Estructura Temporal</h3>
      </div>

      <div className="relative">
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gray-700" />
        <div className="space-y-4 pl-6">
          {sections.map((section, idx) => {
            const startPercent = (section.start / duration) * 100;
            const widthPercent = ((section.end - section.start) / duration) * 100;

            return (
              <div key={idx} className="relative">
                <div className="absolute left-0 top-1/2 transform -translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-purple-500 rounded-full border-2 border-gray-900" />
                <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Music className="w-4 h-4 text-purple-300" />
                      <span className="text-white font-medium capitalize">
                        {section.name || section.type}
                      </span>
                    </div>
                    <span className="text-sm text-gray-400">
                      {formatTime(section.start)} - {formatTime(section.end)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-purple-500 h-2 rounded-full"
                      style={{ width: `${widthPercent}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}


