'use client';

import { useState } from 'react';
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

type SortField = 'popularity' | 'name' | 'duration' | 'date';
type SortOrder = 'asc' | 'desc';

interface SortOptionsProps {
  onSortChange: (field: SortField, order: SortOrder) => void;
  currentField?: SortField;
  currentOrder?: SortOrder;
}

export function SortOptions({ onSortChange, currentField, currentOrder }: SortOptionsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [field, setField] = useState<SortField>(currentField || 'popularity');
  const [order, setOrder] = useState<SortOrder>(currentOrder || 'desc');

  const handleSort = (newField: SortField) => {
    const newOrder = field === newField && order === 'desc' ? 'asc' : 'desc';
    setField(newField);
    setOrder(newOrder);
    onSortChange(newField, newOrder);
  };

  const sortOptions: { field: SortField; label: string }[] = [
    { field: 'popularity', label: 'Popularidad' },
    { field: 'name', label: 'Nombre' },
    { field: 'duration', label: 'Duración' },
    { field: 'date', label: 'Fecha' },
  ];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
      >
        <ArrowUpDown className="w-5 h-5" />
        <span>Ordenar</span>
        {order === 'asc' ? (
          <ArrowUp className="w-4 h-4" />
        ) : (
          <ArrowDown className="w-4 h-4" />
        )}
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-2 w-48 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 p-2 z-50">
          {sortOptions.map((option) => (
            <button
              key={option.field}
              onClick={() => handleSort(option.field)}
              className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                field === option.field
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:bg-white/10'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

