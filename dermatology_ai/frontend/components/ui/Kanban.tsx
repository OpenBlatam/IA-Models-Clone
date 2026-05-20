'use client';

import React, { useState } from 'react';
import { Plus } from 'lucide-react';
import { clsx } from 'clsx';
import { Card } from './Card';
import { Button } from './Button';

interface KanbanCard {
  id: string;
  title: string;
  description?: string;
  status: string;
}

interface KanbanColumn {
  id: string;
  title: string;
  cards: KanbanCard[];
}

interface KanbanProps {
  columns: KanbanColumn[];
  onCardMove?: (cardId: string, fromStatus: string, toStatus: string) => void;
  onCardAdd?: (status: string) => void;
  className?: string;
}

export const Kanban: React.FC<KanbanProps> = ({
  columns,
  onCardMove,
  onCardAdd,
  className,
}) => {
  const [draggedCard, setDraggedCard] = useState<KanbanCard | null>(null);

  const handleDragStart = (card: KanbanCard) => {
    setDraggedCard(card);
  };

  const handleDragOver = (e: React.DragEvent, status: string) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, toStatus: string) => {
    e.preventDefault();
    if (draggedCard && draggedCard.status !== toStatus) {
      onCardMove?.(draggedCard.id, draggedCard.status, toStatus);
    }
    setDraggedCard(null);
  };

  return (
    <div className={clsx('flex gap-4 overflow-x-auto pb-4', className)}>
      {columns.map((column) => (
        <div
          key={column.id}
          className="flex-shrink-0 w-80"
          onDragOver={(e) => handleDragOver(e, column.id)}
          onDrop={(e) => handleDrop(e, column.id)}
        >
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-semibold text-gray-900 dark:text-white">
              {column.title}
            </h3>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {column.cards.length}
            </span>
          </div>
          <div className="space-y-2 min-h-[200px]">
            {column.cards.map((card) => (
              <Card
                key={card.id}
                draggable
                onDragStart={() => handleDragStart(card)}
                className="cursor-move hover:shadow-md transition-shadow"
              >
                <div className="p-4">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                    {card.title}
                  </h4>
                  {card.description && (
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {card.description}
                    </p>
                  )}
                </div>
              </Card>
            ))}
            {onCardAdd && (
              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={() => onCardAdd(column.id)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Agregar tarjeta
              </Button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};


