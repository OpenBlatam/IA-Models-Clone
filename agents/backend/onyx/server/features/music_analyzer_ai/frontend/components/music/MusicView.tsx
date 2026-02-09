'use client';

import { useState } from 'react';
import { Grid, List, Layout } from 'lucide-react';

interface MusicViewProps {
  viewMode: 'grid' | 'list' | 'compact';
  onViewModeChange: (mode: 'grid' | 'list' | 'compact') => void;
}

export function MusicView({ viewMode, onViewModeChange }: MusicViewProps) {
  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Layout className="w-4 h-4 text-purple-300" />
        <h3 className="text-sm font-semibold text-white">Vista</h3>
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => onViewModeChange('grid')}
          className={`p-2 rounded-lg transition-colors ${
            viewMode === 'grid'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
          title="Vista de cuadrícula"
        >
          <Grid className="w-4 h-4" />
        </button>
        <button
          onClick={() => onViewModeChange('list')}
          className={`p-2 rounded-lg transition-colors ${
            viewMode === 'list'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
          title="Vista de lista"
        >
          <List className="w-4 h-4" />
        </button>
        <button
          onClick={() => onViewModeChange('compact')}
          className={`p-2 rounded-lg transition-colors ${
            viewMode === 'compact'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
          title="Vista compacta"
        >
          <List className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}


