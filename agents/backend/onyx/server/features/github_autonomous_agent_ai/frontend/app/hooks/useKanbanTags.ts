import { useState, useEffect, useMemo } from 'react';
import { Task } from '../types/task';

export function useKanbanTags(tasks: Task[]) {
  const [taskTags, setTaskTags] = useState<Map<string, string[]>>(new Map());
  const [tagFilter, setTagFilter] = useState<string>('all');

  // Extraer tags de tareas automáticamente
  useEffect(() => {
    const newTags = new Map<string, string[]>();
    tasks.forEach(task => {
      const tags: string[] = [];
      // Extraer tags del contenido o instrucción
      const text = `${task.instruction} ${task.error || ''} ${task.repository}`.toLowerCase();
      if (text.includes('urgent') || text.includes('urgente')) tags.push('urgent');
      if (text.includes('bug') || text.includes('error')) tags.push('bug');
      if (text.includes('feature') || text.includes('nueva')) tags.push('feature');
      if (text.includes('refactor')) tags.push('refactor');
      if (text.includes('test') || text.includes('prueba')) tags.push('test');
      if (task.status === 'completed') tags.push('done');
      if (task.status === 'failed') tags.push('failed');
      if (task.status === 'processing') tags.push('in-progress');
      newTags.set(task.id, tags);
    });
    setTaskTags(newTags);
  }, [tasks]);

  // Obtener todos los tags únicos
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    taskTags.forEach(tags => tags.forEach(tag => tagSet.add(tag)));
    return Array.from(tagSet).sort();
  }, [taskTags]);

  return {
    taskTags,
    setTaskTags,
    tagFilter,
    setTagFilter,
    allTags,
  };
}

