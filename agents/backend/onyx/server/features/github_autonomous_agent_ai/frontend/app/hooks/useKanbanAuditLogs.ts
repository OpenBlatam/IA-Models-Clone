import { useState, useEffect } from 'react';
import { Task } from '../types/task';

export interface AuditLog {
  id: string;
  action: string;
  details: string;
  timestamp: Date;
}

export function useKanbanAuditLogs(tasks: Task[]) {
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [showAuditLogs, setShowAuditLogs] = useState(false);

  // Logs de auditoría
  useEffect(() => {
    const logAction = (action: string, details: string) => {
      setAuditLogs(prev => [{
        id: `log-${Date.now()}`,
        action,
        details,
        timestamp: new Date()
      }, ...prev].slice(0, 1000)); // Mantener últimos 1000 logs
    };
    
    // Log cuando se completa una tarea
    tasks.filter(t => t.status === 'completed').forEach(task => {
      const existingLog = auditLogs.find(log => log.details.includes(task.id) && log.action === 'Tarea completada');
      if (!existingLog) {
        logAction('Tarea completada', `Tarea ${task.id} completada: ${task.repository}`);
      }
    });
    
    // Log cuando falla una tarea
    tasks.filter(t => t.status === 'failed').forEach(task => {
      const existingLog = auditLogs.find(log => log.details.includes(task.id) && log.action === 'Tarea fallida');
      if (!existingLog) {
        logAction('Tarea fallida', `Tarea ${task.id} falló: ${task.repository} - ${task.error?.substring(0, 50)}`);
      }
    });
  }, [tasks, auditLogs]);

  const logAction = (action: string, details: string) => {
    setAuditLogs(prev => [{
      id: `log-${Date.now()}`,
      action,
      details,
      timestamp: new Date()
    }, ...prev].slice(0, 1000));
  };

  return {
    auditLogs,
    showAuditLogs,
    setShowAuditLogs,
    logAction,
  };
}

