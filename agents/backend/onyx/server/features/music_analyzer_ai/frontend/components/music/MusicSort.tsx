'use client';

import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

interface MusicSortProps {
  onSortChange: (field: string, order: 'asc' | 'desc') => void;
  currentSort?: { field: string; order: 'asc' | 'desc' };
}

export function MusicSort({ onSortChange, currentSort }: MusicSortProps) {
  const sortOptions = [
    { field: 'name', label: 'Nombre' },
    { field: 'popularity', label: 'Popularidad' },
    { field: 'duration', label: 'Duración' },
    { field: 'energy', label: 'Energía' },
    { field: 'danceability', label: 'Bailabilidad' },
    { field: 'tempo', label: 'Tempo' },
    { field: 'date', label: 'Fecha' },
  ];

  const handleSort = (field: string) => {
    if (currentSort?.field === field) {
      onSortChange(field, currentSort.order === 'asc' ? 'desc' : 'asc');
    } else {
      onSortChange(field, 'asc');
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <ArrowUpDown className="w-4 h-4 text-purple-300" />
        <h3 className="text-sm font-semibold text-white">Ordenar por</h3>
      </div>

      <div className="flex flex-wrap gap-2">
        {sortOptions.map((option) => {
          const isActive = currentSort?.field === option.field;
          const Icon = isActive && currentSort.order === 'asc' ? ArrowUp : ArrowDown;

          return (
            <button
              key={option.field}
              onClick={() => handleSort(option.field)}
              className={`px-3 py-1 rounded-lg text-sm transition-colors flex items-center gap-1 ${
                isActive
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {option.label}
              {isActive && <Icon className="w-3 h-3" />}
            </button>
          );
        })}
      </div>
    </div>
  );
}


