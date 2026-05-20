'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { FiChevronDown, FiChevronUp, FiFilter, FiX } from 'react-icons/fi';
import { format } from 'date-fns';
import { getStatusBadge } from '@/utils/status';
import type { TaskListItem } from '@/types/api';

interface TableViewProps {
  tasks: TaskListItem[];
  onTaskClick?: (taskId: string) => void;
  onDelete?: (taskId: string) => void;
}

type SortField = 'created_at' | 'status' | 'query_preview';
type SortDirection = 'asc' | 'desc';

export default function TableView({ tasks, onTaskClick, onDelete }: TableViewProps) {
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());

  const sortedTasks = useMemo(() => {
    return [...tasks].sort((a, b) => {
      let aVal: any, bVal: any;
      
      if (sortField === 'created_at') {
        aVal = new Date(a.created_at).getTime();
        bVal = new Date(b.created_at).getTime();
      } else if (sortField === 'status') {
        aVal = a.status;
        bVal = b.status;
      } else {
        aVal = a.query_preview.toLowerCase();
        bVal = b.query_preview.toLowerCase();
      }

      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });
  }, [tasks, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const toggleRow = (taskId: string) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(taskId)) {
      newSelected.delete(taskId);
    } else {
      newSelected.add(taskId);
    }
    setSelectedRows(newSelected);
  };

  const toggleAll = () => {
    if (selectedRows.size === tasks.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(tasks.map((t) => t.task_id)));
    }
  };

  const renderStatusBadge = (status: string) => {
    const badge = getStatusBadge(status);
    return <span className={`px-2 py-1 rounded text-xs font-medium ${badge.className}`}>{badge.label}</span>;
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? <FiChevronUp size={16} /> : <FiChevronDown size={16} />;
  };

  return (
    <div className="card overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="px-4 py-3 text-left">
              <input
                type="checkbox"
                checked={selectedRows.size === tasks.length && tasks.length > 0}
                onChange={toggleAll}
                className="rounded"
              />
            </th>
            <th
              className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
              onClick={() => handleSort('query_preview')}
            >
              <div className="flex items-center gap-2">
                Consulta
                <SortIcon field="query_preview" />
              </div>
            </th>
            <th
              className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
              onClick={() => handleSort('status')}
            >
              <div className="flex items-center gap-2">
                Estado
                <SortIcon field="status" />
              </div>
            </th>
            <th
              className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
              onClick={() => handleSort('created_at')}
            >
              <div className="flex items-center gap-2">
                Fecha
                <SortIcon field="created_at" />
              </div>
            </th>
            <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">
              ID
            </th>
            <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">
              Acciones
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedTasks.map((task, index) => (
            <motion.tr
              key={task.task_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.02 }}
              className={`border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 ${
                selectedRows.has(task.task_id) ? 'bg-primary-50 dark:bg-primary-900/20' : ''
              }`}
            >
              <td className="px-4 py-3">
                <input
                  type="checkbox"
                  checked={selectedRows.has(task.task_id)}
                  onChange={() => toggleRow(task.task_id)}
                  className="rounded"
                />
              </td>
              <td className="px-4 py-3">
                <div className="max-w-md">
                  <p className="text-sm text-gray-900 dark:text-white truncate">
                    {task.query_preview}
                  </p>
                </div>
              </td>
              <td className="px-4 py-3">{renderStatusBadge(task.status)}</td>
              <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                {format(new Date(task.created_at), 'PPp')}
              </td>
              <td className="px-4 py-3">
                <code className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                  {task.task_id.substring(0, 8)}...
                </code>
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  {task.status === 'completed' && onTaskClick && (
                    <button
                      onClick={() => onTaskClick(task.task_id)}
                      className="btn btn-primary text-xs"
                    >
                      Ver
                    </button>
                  )}
                  {onDelete && (
                    <button
                      onClick={() => onDelete(task.task_id)}
                      className="btn-icon text-red-600"
                      title="Eliminar"
                    >
                      <FiX size={16} />
                    </button>
                  )}
                </div>
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      {selectedRows.size > 0 && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {selectedRows.size} fila(s) seleccionada(s)
          </p>
        </div>
      )}
    </div>
  );
}

