'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api-client';
import type { TaskListItem } from '@/types/api';
import { format } from 'date-fns';
import { showToast } from '@/lib/toast';
import { getStatusBadge } from '@/utils/status';
import DocumentModal from '@/components/DocumentModal';
import SearchBar from '@/components/SearchBar';
import AdvancedFilters from '@/components/AdvancedFilters';
import AdvancedSearch from '@/components/AdvancedSearch';
import QuickFilters from '@/components/QuickFilters';
import Pagination from '@/components/Pagination';
import BatchOperations from '@/components/BatchOperations';
import CalendarView from '@/components/CalendarView';
import ReportGenerator from '@/components/ReportGenerator';
import TableView from '@/components/TableView';
import { useTasks, useSearch } from '@/hooks';
import { motion } from 'framer-motion';
import { FiFilter, FiRefreshCw, FiCalendar, FiFileText, FiList } from 'react-icons/fi';

export default function TasksView() {
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [advancedFilters, setAdvancedFilters] = useState<any>({});
  const [viewMode, setViewMode] = useState<'list' | 'calendar' | 'table'>('list');
  const [searchFilters, setSearchFilters] = useState<any>({});
  const [quickFilters, setQuickFilters] = useState<string[]>([]);

  const {
    tasks: allTasks,
    isLoading,
    isRefreshing,
    currentPage,
    totalPages,
    totalItems,
    setCurrentPage,
    refresh,
    deleteTask,
    cancelTask,
  } = useTasks({ autoRefresh: true, refreshInterval: 30000 });

  const { searchQuery, setSearchQuery, filteredItems } = useSearch(allTasks, {
    fields: ['query_preview', 'task_id'],
  });

  // Apply additional filters
  const tasks = filteredItems.filter((task) => {
    if (filter && task.status !== filter) return false;
    if (quickFilters.length > 0 && !quickFilters.includes(task.status)) return false;
    if (searchFilters.status?.length && !searchFilters.status.includes(task.status)) return false;
    if (searchFilters.dateRange?.start || searchFilters.dateRange?.end) {
      const taskDate = new Date(task.created_at);
      if (searchFilters.dateRange.start && taskDate < searchFilters.dateRange.start) return false;
      if (searchFilters.dateRange.end && taskDate > searchFilters.dateRange.end) return false;
    }
    return true;
  });

  // Quick filter options based on task statuses
  const quickFilterOptions = [
    { id: 'completed', label: 'Completadas', count: allTasks.filter((t) => t.status === 'completed').length },
    { id: 'processing', label: 'Procesando', count: allTasks.filter((t) => t.status === 'processing').length },
    { id: 'failed', label: 'Fallidas', count: allTasks.filter((t) => t.status === 'failed').length },
    { id: 'queued', label: 'En Cola', count: allTasks.filter((t) => t.status === 'queued').length },
  ];

  const handleApplyFilters = (filters: any) => {
    setAdvancedFilters(filters);
  };

  const handleResetFilters = () => {
    setAdvancedFilters({});
    setFilter('');
    setSearchQuery('');
    setQuickFilters([]);
    setSearchFilters({});
  };

  const handleViewDocument = async (taskId: string) => {
    try {
      const task = await apiClient.getTaskStatus(taskId);
      if (task.status === 'completed') {
        setSelectedTaskId(taskId);
      } else {
        showToast('La tarea aún no está completada', 'warning');
      }
    } catch (error: any) {
      showToast(error.message || 'Error al cargar documento', 'error');
    }
  };

  const renderStatusBadge = (status: string) => {
    const badge = getStatusBadge(status);
    return <span className={`badge ${badge.className}`}>{badge.label}</span>;
  };

  return (
    <div>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Gestión de Tareas</h2>
            <p className="text-gray-600">Monitorea y gestiona todas tus tareas de generación</p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <SearchBar
              placeholder="Buscar tareas por ID o consulta..."
              onSearch={setSearchQuery}
            />
          </div>
          <QuickFilters
            filters={quickFilterOptions}
            selected={quickFilters}
            onChange={setQuickFilters}
          />
          <div className="flex items-center gap-3">
            <AdvancedFilters onApply={handleApplyFilters} onReset={handleResetFilters} />
            <div className="relative">
              <FiFilter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="select pl-10 pr-4"
              >
                <option value="">Todas</option>
                <option value="queued">En Cola</option>
                <option value="processing">Procesando</option>
                <option value="completed">Completadas</option>
                <option value="failed">Fallidas</option>
              </select>
            </div>
            <button
              onClick={refresh}
              className="btn btn-secondary"
              disabled={isLoading || isRefreshing}
            >
              <FiRefreshCw className={isRefreshing ? 'animate-spin' : ''} size={16} />
              Actualizar
            </button>
            <AdvancedSearch
              onSearch={(filters) => setSearchFilters(filters)}
              onReset={() => setSearchFilters({})}
            />
            <div className="flex gap-1">
              <button
                onClick={() => setViewMode('list')}
                className={`btn btn-secondary ${viewMode === 'list' ? 'bg-primary-50 dark:bg-primary-900/30' : ''}`}
                title="Vista de lista"
              >
                <FiList size={16} />
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={`btn btn-secondary ${viewMode === 'table' ? 'bg-primary-50 dark:bg-primary-900/30' : ''}`}
                title="Vista de tabla"
              >
                <FiFileText size={16} />
              </button>
              <button
                onClick={() => setViewMode('calendar')}
                className={`btn btn-secondary ${viewMode === 'calendar' ? 'bg-primary-50 dark:bg-primary-900/30' : ''}`}
                title="Vista de calendario"
              >
                <FiCalendar size={16} />
              </button>
            </div>
            <ReportGenerator />
          </div>
        </div>
      </motion.div>

      {viewMode === 'calendar' ? (
        <CalendarView
          onTaskClick={(taskId) => handleViewDocument(taskId)}
        />
      ) : viewMode === 'table' ? (
        <TableView
          tasks={tasks}
          onTaskClick={(taskId) => handleViewDocument(taskId)}
          onDelete={deleteTask}
        />
      ) : (
        <>

      {isLoading ? (
        <div className="card text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Cargando tareas...</p>
        </div>
      ) : tasks.length === 0 ? (
        <div className="card text-center py-12">
          <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M9 11l3 3L22 4" />
            <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
          </svg>
          <p className="text-gray-600">No hay tareas disponibles</p>
        </div>
      ) : (
        <div className="space-y-4">
          {tasks.map((task, index) => (
            <motion.div
              key={task.task_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="card hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="font-mono text-sm text-gray-500">{task.task_id}</span>
                    {renderStatusBadge(task.status)}
                  </div>
                  <p className="text-gray-700 mb-3">{task.query_preview}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-500">
                    <span>
                      Creada: {format(new Date(task.created_at), "PPp")}
                    </span>
                    {task.status === 'processing' && (
                      <span className="text-primary-600">Procesando...</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  {task.status === 'completed' && (
                    <button
                      onClick={() => handleViewDocument(task.task_id)}
                      className="btn btn-primary text-sm"
                    >
                      Ver Documento
                    </button>
                  )}
                  {(task.status === 'queued' || task.status === 'processing') && (
                    <button
                      onClick={() => cancelTask(task.task_id)}
                      className="btn btn-secondary text-sm"
                    >
                      Cancelar
                    </button>
                  )}
                  <button
                    onClick={() => deleteTask(task.task_id)}
                    className="btn-icon text-red-600 hover:bg-red-50"
                    title="Eliminar"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 6h18" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {tasks.length > 0 && totalPages > 1 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={(page) => {
            setCurrentPage(page);
          }}
          itemsPerPage={20}
          totalItems={totalItems}
        />
      )}

      <BatchOperations
        items={tasks}
        onDelete={async (ids) => {
          for (const id of ids) {
            try {
              await apiClient.deleteTask(id);
            } catch (error) {
              console.error(`Error deleting task ${id}:`, error);
            }
          }
          refresh();
        }}
        getId={(task) => task.task_id}
        getLabel={(task) => task.query_preview}
      />

      {selectedTaskId && (
        <DocumentModal taskId={selectedTaskId} onClose={() => setSelectedTaskId(null)} />
      )}
        </>
      )}
    </div>
  );
}

