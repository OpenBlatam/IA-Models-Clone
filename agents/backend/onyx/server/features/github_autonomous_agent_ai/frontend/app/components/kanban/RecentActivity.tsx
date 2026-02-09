'use client';

import { Task } from '../../types/task';
import { cn } from '../../utils/cn';

interface Activity {
  task: Task;
  action: string;
  timestamp: Date;
}

interface RecentActivityProps {
  recentActivity: Activity[];
  onSelectTask: (task: Task) => void;
  showRecentActivity: boolean;
}

export function RecentActivity({ recentActivity, onSelectTask, showRecentActivity }: RecentActivityProps) {
  if (!showRecentActivity) return null;

  return (
    <div className="px-4 py-3 bg-white border-b border-gray-200 max-h-96 overflow-y-auto">
      <h3 className="font-semibold text-sm text-gray-900 mb-3">Actividad Reciente</h3>
      {recentActivity.length === 0 ? (
        <p className="text-sm text-gray-500 text-center py-4">No hay actividad reciente</p>
      ) : (
        <div className="space-y-2">
          {recentActivity.map((activity, index) => {
            const timeAgo = (() => {
              const diffMs = Date.now() - activity.timestamp.getTime();
              const diffMins = Math.floor(diffMs / 1000 / 60);
              if (diffMins < 1) return 'ahora';
              if (diffMins < 60) return `hace ${diffMins}m`;
              const diffHours = Math.floor(diffMins / 60);
              if (diffHours < 24) return `hace ${diffHours}h`;
              const diffDays = Math.floor(diffHours / 24);
              return `hace ${diffDays}d`;
            })();
            
            return (
              <div
                key={`${activity.task.id}-${index}`}
                className="flex items-start gap-3 p-2 hover:bg-gray-50 rounded-lg cursor-pointer transition-colors"
                onClick={() => onSelectTask(activity.task)}
              >
                <div className={cn(
                  "w-2 h-2 rounded-full mt-2 flex-shrink-0",
                  activity.action.includes('completada') ? "bg-green-500" :
                  activity.action.includes('creada') ? "bg-blue-500" :
                  "bg-gray-400"
                )} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-gray-600 truncate">
                      {activity.task.repository}
                    </span>
                    <span className="text-xs text-gray-400">{timeAgo}</span>
                  </div>
                  <p className="text-sm text-gray-900 line-clamp-1">
                    Tarea {activity.action}: {activity.task.instruction.substring(0, 60)}...
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

