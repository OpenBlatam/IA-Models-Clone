'use client';

import { Task } from '../../types/task';
import { cn } from '../../utils/cn';

interface CalendarViewProps {
  tasks: Task[];
  onSelectTask: (task: Task) => void;
  month?: number | null;
  year?: number | null;
  onMonthChange?: (month: number, year: number) => void;
}

export function CalendarView({ tasks, onSelectTask, month, year, onMonthChange }: CalendarViewProps) {
  const now = new Date();
  const currentMonth = month !== null && month !== undefined ? month : now.getMonth();
  const currentYear = year !== null && year !== undefined ? year : now.getFullYear();
  const firstDay = new Date(currentYear, currentMonth, 1);
  const lastDay = new Date(currentYear, currentMonth + 1, 0);
  const startDay = firstDay.getDay();
  const daysInMonth = lastDay.getDate();
  
  const handlePreviousMonth = () => {
    const newDate = new Date(currentYear, currentMonth - 1, 1);
    onMonthChange?.(newDate.getMonth(), newDate.getFullYear());
  };
  
  const handleNextMonth = () => {
    const newDate = new Date(currentYear, currentMonth + 1, 1);
    onMonthChange?.(newDate.getMonth(), newDate.getFullYear());
  };
  
  const handleToday = () => {
    const today = new Date();
    onMonthChange?.(today.getMonth(), today.getFullYear());
  };

  const calendarDays = [];
  
  // Días vacíos al inicio
  for (let i = 0; i < startDay; i++) {
    calendarDays.push(<div key={`empty-${i}`} className="h-24"></div>);
  }
  
  // Días del mes
  for (let day = 1; day <= daysInMonth; day++) {
    const date = new Date(currentYear, currentMonth, day);
    const dayTasks = tasks.filter(task => {
      const taskDate = new Date(task.createdAt);
      return taskDate.getDate() === day && 
             taskDate.getMonth() === currentMonth && 
             taskDate.getFullYear() === currentYear;
    });
    
    const isToday = day === now.getDate() && 
                   currentMonth === now.getMonth() && 
                   currentYear === now.getFullYear();
    
    calendarDays.push(
      <div
        key={day}
        className={cn(
          "h-24 border border-gray-200 rounded-lg p-2 overflow-y-auto",
          isToday ? "bg-blue-50 border-blue-300" : "bg-white hover:bg-gray-50"
        )}
      >
        <div className={cn(
          "text-sm font-medium mb-1",
          isToday ? "text-blue-700" : "text-gray-900"
        )}>
          {day}
        </div>
        <div className="space-y-1">
          {dayTasks.slice(0, 3).map(task => (
            <div
              key={task.id}
              onClick={() => onSelectTask(task)}
              className={cn(
                "text-xs px-1.5 py-0.5 rounded cursor-pointer truncate",
                task.status === 'completed' ? "bg-green-100 text-green-800" :
                task.status === 'failed' ? "bg-red-100 text-red-800" :
                task.status === 'processing' || task.status === 'running' ? "bg-blue-100 text-blue-800" :
                "bg-gray-100 text-gray-800"
              )}
              title={task.instruction}
            >
              {task.instruction.substring(0, 20)}...
            </div>
          ))}
          {dayTasks.length > 3 && (
            <div className="text-xs text-gray-500 px-1.5">
              +{dayTasks.length - 3} más
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Calendario de Tareas</h2>
          {onMonthChange && (
            <div className="flex items-center gap-2">
              <button
                onClick={handlePreviousMonth}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                title="Mes anterior"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span className="text-sm font-medium text-gray-700 min-w-[150px] text-center">
                {new Date(currentYear, currentMonth, 1).toLocaleDateString('es-ES', { month: 'long', year: 'numeric' })}
              </span>
              <button
                onClick={handleNextMonth}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                title="Mes siguiente"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <button
                onClick={handleToday}
                className="px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                title="Ir a hoy"
              >
                Hoy
              </button>
            </div>
          )}
        </div>
        <div className="grid grid-cols-7 gap-2">
          {/* Días de la semana */}
          {['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'].map(day => (
            <div key={day} className="text-center text-sm font-medium text-gray-600 py-2">
              {day}
            </div>
          ))}
          
          {/* Días del mes */}
          {calendarDays}
        </div>
      </div>
    </div>
  );
}

