'use client';

import { Task } from '../../types/task';

interface RepoSummaryProps {
  repositories: string[];
  tasks: Task[];
  onSelectRepository: (repo: string) => void;
  showRepoSummary: boolean;
}

export function RepoSummary({ repositories, tasks, onSelectRepository, showRepoSummary }: RepoSummaryProps) {
  if (!showRepoSummary) return null;

  return (
    <div className="px-4 py-3 bg-white border-b border-gray-200">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {repositories.map((repo) => {
          const repoTasks = tasks.filter(t => t.repository === repo);
          const repoStats = {
            total: repoTasks.length,
            completed: repoTasks.filter(t => t.status === 'completed').length,
            processing: repoTasks.filter(t => t.status === 'processing' || t.status === 'running').length,
            failed: repoTasks.filter(t => t.status === 'failed').length,
            withCommits: repoTasks.filter(t => t.executionResult?.commitSha).length,
          };
          return (
            <div key={repo} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-sm text-gray-900 mb-2 truncate">{repo}</h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-600">Total:</span>
                  <span className="ml-1 font-semibold">{repoStats.total}</span>
                </div>
                <div>
                  <span className="text-green-600">✓ Completadas:</span>
                  <span className="ml-1 font-semibold">{repoStats.completed}</span>
                </div>
                <div>
                  <span className="text-blue-600">🔄 Procesando:</span>
                  <span className="ml-1 font-semibold">{repoStats.processing}</span>
                </div>
                <div>
                  <span className="text-red-600">✗ Fallidas:</span>
                  <span className="ml-1 font-semibold">{repoStats.failed}</span>
                </div>
                <div className="col-span-2">
                  <span className="text-gray-600">Commits:</span>
                  <span className="ml-1 font-semibold">{repoStats.withCommits}</span>
                </div>
              </div>
              <button
                onClick={() => onSelectRepository(repo)}
                className="mt-2 w-full text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 px-2 py-1 rounded transition-colors"
              >
                Ver tareas →
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

