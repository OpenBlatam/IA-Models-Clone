import { useState, useCallback } from 'react';
import { DragEndEvent } from '@dnd-kit/core';
import { Task, TaskStatus } from '../types/task';
import { KANBAN_COLUMNS } from '../constants/task-constants';
import { toast } from 'sonner';

interface UseKanbanDragDropProps {
  tasks: Task[];
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  onActivityUpdate?: (activity: { task: Task; action: string; timestamp: Date }) => void;
}

export function useKanbanDragDrop({
  tasks,
  updateTask,
  onActivityUpdate,
}: UseKanbanDragDropProps) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [dragOverColumn, setDragOverColumn] = useState<string | null>(null);

  const handleDragStart = useCallback((event: any) => {
    setActiveId(event.active.id);
  }, []);

  const handleDragOver = useCallback((event: any) => {
    if (event.over) {
      setDragOverColumn(event.over.id as string);
    }
  }, []);

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event;

    setDragOverColumn(null);

    if (!over || active.id === over.id) {
      setActiveId(null);
      return;
    }

    const activeTask = tasks.find((t) => t.id === active.id);
    if (!activeTask) {
      setActiveId(null);
      return;
    }

    // Obtener el nuevo status desde el over.id (que es el columnId)
    const newStatus = over.id as TaskStatus;

    if (newStatus && newStatus !== activeTask.status) {
      updateTask(activeTask.id, { status: newStatus });
      const column = KANBAN_COLUMNS.find((col) => col.id === newStatus);

      // Agregar a actividad reciente
      if (onActivityUpdate) {
        onActivityUpdate({
          task: { ...activeTask, status: newStatus },
          action: `movida a ${column?.label || newStatus}`,
          timestamp: new Date(),
        });
      }

      toast.success('Estado actualizado', {
        description: `Tarea movida a ${column?.label || newStatus}`,
        duration: 3000,
      });
    }

    setActiveId(null);
  }, [tasks, updateTask, onActivityUpdate]);

  const handleDragCancel = useCallback(() => {
    setActiveId(null);
    setDragOverColumn(null);
  }, []);

  return {
    activeId,
    dragOverColumn,
    handleDragStart,
    handleDragOver,
    handleDragEnd,
    handleDragCancel,
  };
}

