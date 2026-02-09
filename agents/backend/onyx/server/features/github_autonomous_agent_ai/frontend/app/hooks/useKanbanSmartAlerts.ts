import { useState, useEffect } from 'react';
import { Task } from '../types/task';

export interface SmartAlert {
  id: string;
  type: string;
  message: string;
  timestamp: Date;
}

export function useKanbanSmartAlerts(tasks: Task[]) {
  const [smartAlerts, setSmartAlerts] = useState<SmartAlert[]>([]);
  const [showSmartAlerts, setShowSmartAlerts] = useState(false);

  // Alertas inteligentes
  useEffect(() => {
    const newAlerts: SmartAlert[] = [];
    
    // Alerta: Tasa de éxito baja
    const successRate = tasks.length > 0 
      ? (tasks.filter(t => t.status === 'completed').length / tasks.length) * 100 
      : 100;
    if (successRate < 50 && tasks.length > 10) {
      newAlerts.push({
        id: 'low-success-rate',
        type: 'warning',
        message: `Tasa de éxito baja: ${Math.round(successRate)}%. Considera revisar las tareas fallidas.`,
        timestamp: new Date()
      });
    }
    
    // Alerta: Muchas tareas en proceso
    const processingCount = tasks.filter(t => t.status === 'processing' || t.status === 'running').length;
    if (processingCount > 20) {
      newAlerts.push({
        id: 'many-processing',
        type: 'info',
        message: `${processingCount} tareas en proceso. El sistema está muy activo.`,
        timestamp: new Date()
      });
    }
    
    // Alerta: Tareas fallidas recientes
    const recentFailures = tasks.filter(t => 
      t.status === 'failed' && 
      new Date(t.updatedAt || t.createdAt).getTime() > Date.now() - 3600000
    );
    if (recentFailures.length > 5) {
      newAlerts.push({
        id: 'recent-failures',
        type: 'error',
        message: `${recentFailures.length} tareas fallaron en la última hora.`,
        timestamp: new Date()
      });
    }
    
    // Alerta: Sin actividad reciente
    const lastActivity = tasks.length > 0 
      ? Math.max(...tasks.map(t => new Date(t.updatedAt || t.createdAt).getTime()))
      : 0;
    const hoursSinceActivity = (Date.now() - lastActivity) / (1000 * 60 * 60);
    if (hoursSinceActivity > 24 && tasks.length > 0) {
      newAlerts.push({
        id: 'no-recent-activity',
        type: 'warning',
        message: `Sin actividad en las últimas ${Math.round(hoursSinceActivity)} horas.`,
        timestamp: new Date()
      });
    }
    
    setSmartAlerts(newAlerts);
  }, [tasks]);

  return {
    smartAlerts,
    showSmartAlerts,
    setShowSmartAlerts,
  };
}

