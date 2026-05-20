'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiCalendar, FiChevronLeft, FiChevronRight, FiClock } from 'react-icons/fi';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameDay, isToday, startOfWeek, endOfWeek } from 'date-fns';
import { apiClient } from '@/lib/api-client';
import { getStatusBadge } from '@/utils/status';
import StatusBadge from '@/components/StatusBadge';
import type { TaskListItem } from '@/types/api';

interface CalendarViewProps {
  onTaskClick?: (taskId: string) => void;
}

export default function CalendarView({ onTaskClick }: CalendarViewProps) {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);

  useEffect(() => {
    loadTasks();
  }, [currentDate]);

  const loadTasks = async () => {
    try {
      const response = await apiClient.listTasks({ limit: 1000 });
      setTasks(response.tasks);
    } catch (error) {
      console.error('Error loading tasks:', error);
    }
  };

  const monthStart = startOfMonth(currentDate);
  const monthEnd = endOfMonth(currentDate);
  const calendarStart = startOfWeek(monthStart, { weekStartsOn: 1 });
  const calendarEnd = endOfWeek(monthEnd, { weekStartsOn: 1 });
  const days = eachDayOfInterval({ start: calendarStart, end: calendarEnd });

  const getTasksForDate = (date: Date) => {
    return tasks.filter((task) => {
      const taskDate = new Date(task.created_at);
      return isSameDay(taskDate, date);
    });
  };

  const navigateMonth = (direction: 'prev' | 'next') => {
    setCurrentDate((prev) => {
      const newDate = new Date(prev);
      if (direction === 'prev') {
        newDate.setMonth(prev.getMonth() - 1);
      } else {
        newDate.setMonth(prev.getMonth() + 1);
      }
      return newDate;
    });
  };

  const weekDays = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <FiCalendar size={24} className="text-primary-600" />
          <h3 className="text-xl font-bold text-gray-900 dark:text-white">
            Vista de Calendario
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigateMonth('prev')}
            className="btn-icon"
            title="Mes anterior"
          >
            <FiChevronLeft size={20} />
          </button>
          <span className="text-lg font-semibold text-gray-900 dark:text-white min-w-[200px] text-center">
            {format(currentDate, 'MMMM yyyy')}
          </span>
          <button
            onClick={() => navigateMonth('next')}
            className="btn-icon"
            title="Mes siguiente"
          >
            <FiChevronRight size={20} />
          </button>
        </div>
      </div>

      {/* Week Days Header */}
      <div className="grid grid-cols-7 gap-1 mb-2">
        {weekDays.map((day) => (
          <div
            key={day}
            className="text-center text-sm font-medium text-gray-500 dark:text-gray-400 py-2"
          >
            {day}
          </div>
        ))}
      </div>

      {/* Calendar Grid */}
      <div className="grid grid-cols-7 gap-1">
        {days.map((day, index) => {
          const dayTasks = getTasksForDate(day);
          const isCurrentMonth = day.getMonth() === currentDate.getMonth();
          const isSelected = selectedDate && isSameDay(day, selectedDate);
          const isTodayDate = isToday(day);

          return (
            <motion.button
              key={day.toISOString()}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: index * 0.01 }}
              onClick={() => setSelectedDate(day)}
              className={`min-h-[80px] p-2 border rounded-lg transition-all ${
                !isCurrentMonth
                  ? 'opacity-30 bg-gray-50 dark:bg-gray-800'
                  : isSelected
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700'
              } ${isTodayDate ? 'ring-2 ring-primary-500' : ''}`}
            >
              <div
                className={`text-sm font-medium mb-1 ${
                  isTodayDate
                    ? 'text-primary-600 dark:text-primary-400'
                    : isCurrentMonth
                    ? 'text-gray-900 dark:text-white'
                    : 'text-gray-400 dark:text-gray-600'
                }`}
              >
                {format(day, 'd')}
              </div>
              <div className="space-y-1">
                {dayTasks.slice(0, 3).map((task) => {
                  const statusColor = getStatusBadge(task.status).className.includes('green') 
                    ? 'bg-green-100 dark:bg-green-900' 
                    : getStatusBadge(task.status).className.includes('yellow')
                    ? 'bg-yellow-100 dark:bg-yellow-900'
                    : getStatusBadge(task.status).className.includes('red')
                    ? 'bg-red-100 dark:bg-red-900'
                    : 'bg-blue-100 dark:bg-blue-900';
                  
                  return (
                    <div
                      key={task.task_id}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (onTaskClick) onTaskClick(task.task_id);
                      }}
                      className={`text-xs p-1 rounded truncate cursor-pointer ${statusColor} ${
                        task.status === 'completed'
                          ? 'text-green-800 dark:text-green-200'
                          : task.status === 'processing'
                          ? 'text-yellow-800 dark:text-yellow-200'
                          : task.status === 'failed'
                          ? 'text-red-800 dark:text-red-200'
                          : 'text-blue-800 dark:text-blue-200'
                      }`}
                      title={task.query_preview}
                    >
                      {task.query_preview.substring(0, 15)}...
                    </div>
                  );
                })}
                {dayTasks.length > 3 && (
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    +{dayTasks.length - 3} más
                  </div>
                )}
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Selected Date Tasks */}
      {selectedDate && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg"
        >
          <div className="flex items-center gap-2 mb-3">
            <FiClock size={18} className="text-primary-600" />
            <h4 className="font-semibold text-gray-900 dark:text-white">
              Tareas del {format(selectedDate, 'PP')}
            </h4>
          </div>
          {getTasksForDate(selectedDate).length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              No hay tareas para esta fecha
            </p>
          ) : (
            <div className="space-y-2">
              {getTasksForDate(selectedDate).map((task) => (
                <div
                  key={task.task_id}
                  onClick={() => onTaskClick?.(task.task_id)}
                  className="p-3 bg-white dark:bg-gray-800 rounded-lg cursor-pointer hover:shadow-md transition-shadow"
                >
                  <p className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                    {task.query_preview}
                  </p>
                  <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                    <span className="font-mono">{task.task_id}</span>
                    <span>•</span>
                    <StatusBadge status={task.status} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}

