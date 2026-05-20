'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  DndContext,
  DragOverlay,
  closestCenter,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { Task } from '../../types/task';
import { KANBAN_COLUMNS } from '../../constants/task-constants';
import { getStatusColor } from '../../utils/task-helpers';
import { cn } from '../../utils/cn';
import { SortableTaskCard } from './SortableTaskCard';

interface KanbanBoardProps {
  tasksByColumn: Record<string, Task[]>;
  cardSize: 'compact' | 'normal' | 'expanded';
  groupByRepo: boolean;
  groupByModel: boolean;
  activeId: string | null;
  dragOverColumn: string | null;
  onDragStart: (event: any) => void;
  onDragOver: (event: any) => void;
  onDragEnd: (event: DragEndEvent) => void;
  onSelectTask: (task: Task) => void;
  onDeleteTask: (taskId: string) => void;
  onQuickView: (task: Task | null) => void;
  allTasks: Task[];
}

export function KanbanBoard({
  tasksByColumn,
  cardSize,
  groupByRepo,
  groupByModel,
  activeId,
  dragOverColumn,
  onDragStart,
  onDragOver,
  onDragEnd,
  onSelectTask,
  onDeleteTask,
  onQuickView,
  allTasks,
}: KanbanBoardProps) {
  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragEnd={onDragEnd}
      onDragCancel={() => {}}
    >
      <div className="flex gap-4 min-w-max">
        {KANBAN_COLUMNS.map((column) => {
          const columnTasks = tasksByColumn[column.id] || [];
          return (
            <motion.div
              key={column.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              className={cn(
                "flex-shrink-0 bg-white rounded-lg shadow-sm border flex flex-col",
                cardSize === 'compact' ? 'w-64' :
                cardSize === 'expanded' ? 'w-96' :
                'w-80',
                column.id === 'processing' ? 'border-yellow-200' :
                column.id === 'running' ? 'border-blue-200' :
                column.id === 'completed' ? 'border-green-200' :
                column.id === 'failed' ? 'border-red-200' :
                column.id === 'stopped' ? 'border-orange-200' :
                'border-gray-200'
              )}
            >
              {/* Column Header */}
              <div
                id={column.id}
                className={cn(
                  "p-4 border-b-2 sticky top-0 z-10 bg-white",
                  getStatusColor(column.id)
                )}
                style={{
                  backgroundImage: column.id === 'processing' 
                    ? 'linear-gradient(to bottom, rgba(251, 191, 36, 0.05), rgba(251, 191, 36, 0.05))'
                    : column.id === 'running'
                    ? 'linear-gradient(to bottom, rgba(59, 130, 246, 0.05), rgba(59, 130, 246, 0.05))'
                    : column.id === 'completed'
                    ? 'linear-gradient(to bottom, rgba(34, 197, 94, 0.05), rgba(34, 197, 94, 0.05))'
                    : column.id === 'failed'
                    ? 'linear-gradient(to bottom, rgba(239, 68, 68, 0.05), rgba(239, 68, 68, 0.05))'
                    : 'none'
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <div
                      className={cn(
                        "h-2 w-2 rounded-full flex-shrink-0",
                        column.id === 'processing' ? 'bg-yellow-500' :
                        column.id === 'running' ? 'bg-blue-500' :
                        column.id === 'completed' ? 'bg-green-500' :
                        column.id === 'failed' ? 'bg-red-500' :
                        column.id === 'stopped' ? 'bg-orange-500' :
                        'bg-gray-400'
                      )}
                    />
                    <h2 className="font-semibold text-sm text-gray-900 truncate">
                      {column.label}
                    </h2>
                  </div>
                  <span className={cn(
                    "px-2.5 py-1 rounded-full text-xs font-semibold flex-shrink-0",
                    column.id === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                    column.id === 'running' ? 'bg-blue-100 text-blue-800' :
                    column.id === 'completed' ? 'bg-green-100 text-green-800' :
                    column.id === 'failed' ? 'bg-red-100 text-red-800' :
                    column.id === 'stopped' ? 'bg-orange-100 text-orange-800' :
                    'bg-gray-100 text-gray-800'
                  )}>
                    {columnTasks.length}
                  </span>
                  <Link
                    href="/agent-control"
                    onClick={(e) => e.stopPropagation()}
                    className="ml-2 p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors flex-shrink-0"
                    title="Agregar nueva tarea"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </Link>
                </div>
              </div>

              {/* Tasks */}
              <SortableContext
                items={columnTasks.map((t) => t.id)}
                strategy={verticalListSortingStrategy}
              >
                <div className="p-2 space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto flex-1 min-h-[200px]">
                  {columnTasks.length === 0 ? (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex flex-col items-center justify-center py-12 text-gray-400"
                    >
                      <svg className="w-12 h-12 mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <p className="text-sm font-medium">No hay tareas</p>
                      <p className="text-xs mt-1 opacity-75">Arrastra tareas aquí</p>
                    </motion.div>
                  ) : (
                    (() => {
                      let tasksToShow = columnTasks;
                      
                      if (groupByRepo) {
                        tasksToShow = [...columnTasks].sort((a, b) => a.repository.localeCompare(b.repository));
                      } else if (groupByModel) {
                        tasksToShow = [...columnTasks].sort((a, b) => {
                          const modelA = a.model || 'Sin modelo';
                          const modelB = b.model || 'Sin modelo';
                          return modelA.localeCompare(modelB);
                        });
                      }
                      
                      let currentRepo = '';
                      let currentModel = '';
                      
                      return tasksToShow.map((task) => {
                        const showRepoHeader = groupByRepo && task.repository !== currentRepo;
                        const showModelHeader = groupByModel && (task.model || 'Sin modelo') !== currentModel;
                        
                        if (showRepoHeader) currentRepo = task.repository;
                        if (showModelHeader) currentModel = task.model || 'Sin modelo';
                        
                        return (
                          <div key={task.id}>
                            {showRepoHeader && (
                              <div className="px-2 py-1.5 mb-1 bg-blue-50 border border-blue-200 rounded text-xs font-medium text-blue-700 sticky top-0 z-10 flex items-center gap-1">
                                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                </svg>
                                {task.repository}
                              </div>
                            )}
                            {showModelHeader && (
                              <div className="px-2 py-1.5 mb-1 bg-purple-50 border border-purple-200 rounded text-xs font-medium text-purple-700 sticky top-0 z-10 flex items-center gap-1">
                                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                                {task.model || 'Sin modelo'}
                              </div>
                            )}
                            <SortableTaskCard 
                              task={task} 
                              onDelete={onDeleteTask}
                              onSelect={onSelectTask}
                              cardSize={cardSize}
                              onQuickView={onQuickView}
                            />
                          </div>
                        );
                      });
                    })()
                  )}
                </div>
              </SortableContext>
            </motion.div>
          );
        })}
      </div>
      <DragOverlay>
        {activeId ? (() => {
          const activeTask = allTasks.find((t) => t.id === activeId);
          if (!activeTask) return null;
          
          const isProcessing = activeTask.status === 'processing' || activeTask.status === 'running';
          const isCompleted = activeTask.status === 'completed';
          const isFailed = activeTask.status === 'failed';
          const hasError = !!activeTask.error;
          
          return (
            <div className={cn(
              "p-3 bg-white border-2 rounded-lg shadow-2xl opacity-95 max-w-xs transform rotate-2",
              isFailed || hasError ? "border-red-400" :
              isCompleted ? "border-green-400" :
              isProcessing ? "border-blue-400" :
              "border-gray-400"
            )}>
              <div className="flex items-center gap-2 mb-2">
                <p className="text-xs font-medium text-gray-600 truncate">
                  {activeTask.repository}
                </p>
                {isProcessing && (
                  <svg className="w-3 h-3 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                )}
                {isCompleted && !hasError && (
                  <svg className="w-3 h-3 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                )}
                {(isFailed || hasError) && (
                  <svg className="w-3 h-3 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                )}
              </div>
              <p className="text-sm font-medium text-gray-800 line-clamp-2">
                {activeTask.instruction}
              </p>
              {activeTask.executionResult?.commitSha && (
                <div className="mt-2 flex items-center gap-1 text-xs text-purple-600 bg-purple-50 px-2 py-1 rounded">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="font-mono">{activeTask.executionResult.commitSha.substring(0, 7)}</span>
                </div>
              )}
            </div>
          );
        })() : null}
      </DragOverlay>
    </DndContext>
  );
}

