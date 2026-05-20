import React from 'react';
import { useTasks } from '../hooks/useAPI';
import { Button } from '../components/ui/Button';
import { StatusBadge } from '../components/ui/StatusBadge';

type Page = 'main' | 'agent-control' | 'kanban' | 'continuous-agent';

interface KanbanPageProps {
  onNavigate: (page: Page) => void;
}

const KanbanPage: React.FC<KanbanPageProps> = ({ onNavigate }) => {
  const { tasks, loading, error, createTask, updateTask, deleteTask } = useTasks();

  const tasksByStatus = {
    pending: tasks.filter((t) => t.status === 'pending'),
    in_progress: tasks.filter((t) => t.status === 'in_progress'),
    completed: tasks.filter((t) => t.status === 'completed'),
    failed: tasks.filter((t) => t.status === 'failed'),
  };

  const statusLabels = {
    pending: 'Pendiente',
    in_progress: 'En Progreso',
    completed: 'Completado',
    failed: 'Fallido',
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-10 md:py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-normal text-black">Kanban Board</h1>
            <p className="text-sm text-gray-500 mt-2">
              Visualiza y gestiona tus tareas en un tablero Kanban
            </p>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black"></div>
          </div>
        ) : error ? (
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="text-center py-8 text-red-600">
              <p>{error}</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(tasksByStatus).map(([status, statusTasks]) => (
              <div key={status} className="flex flex-col">
                <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                  <div className="mb-4">
                    <h3 className="text-black text-lg font-normal mb-1">{statusLabels[status as keyof typeof statusLabels]}</h3>
                    <p className="text-sm text-gray-500">{`${statusTasks.length} tarea(s)`}</p>
                  </div>
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {statusTasks.length === 0 ? (
                      <p className="text-sm text-gray-400 text-center py-4">
                        No hay tareas
                      </p>
                    ) : (
                      statusTasks.map((task) => (
                        <div
                          key={task.id}
                          className="bg-white border border-gray-200 rounded-lg p-3 hover:shadow-sm transition-shadow cursor-pointer"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <h4 className="font-medium text-sm flex-1 text-black">{task.title}</h4>
                            <StatusBadge status={task.status} />
                          </div>
                          {task.description && (
                            <p className="text-xs text-gray-500 line-clamp-2 mb-2">
                              {task.description}
                            </p>
                          )}
                          <div className="flex justify-between items-center mt-2 pt-2 border-t border-gray-100">
                            <span className="text-xs text-gray-400">
                              {new Date(task.created_at).toLocaleDateString()}
                            </span>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => deleteTask(task.id)}
                            >
                              Eliminar
                            </Button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default KanbanPage;
