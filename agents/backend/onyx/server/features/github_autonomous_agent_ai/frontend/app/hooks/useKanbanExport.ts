import { useCallback } from 'react';
import { Task } from '../types/task';

export function useKanbanExport() {
  const handleExportTasks = useCallback((tasks: Task[]) => {
    const dataStr = JSON.stringify(tasks, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `kanban-tasks-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  const handleExportToCSV = useCallback((tasks: Task[]) => {
    if (tasks.length === 0) {
      return;
    }

    // Encabezados CSV
    const headers = [
      'ID',
      'Repositorio',
      'Instrucción',
      'Estado',
      'Modelo',
      'Creada',
      'Inicio Procesamiento',
      'Completada',
      'Error',
      'Commit SHA',
      'Branch',
      'Commit URL',
    ];

    // Convertir tareas a filas CSV
    const rows = tasks.map((task) => {
      const row = [
        task.id,
        task.repository,
        `"${(task.instruction || '').replace(/"/g, '""')}"`, // Escapar comillas
        task.status,
        task.model || '',
        task.createdAt ? new Date(task.createdAt).toISOString() : '',
        task.processingStartedAt ? new Date(task.processingStartedAt).toISOString() : '',
        task.status === 'completed' ? 'Sí' : 'No',
        task.error ? `"${task.error.replace(/"/g, '""')}"` : '',
        task.executionResult?.commitSha || '',
        task.executionResult?.branch || '',
        task.executionResult?.commitUrl || '',
      ];
      return row.join(',');
    });

    // Crear contenido CSV con BOM para UTF-8
    const csvContent = '\uFEFF' + [headers.join(','), ...rows].join('\n');

    // Crear blob y descargar
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `kanban-tasks-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  return {
    handleExportTasks,
    handleExportToCSV,
  };
}

