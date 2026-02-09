import { useState, useMemo, useEffect } from 'react';
import { useLocalStorage } from './useLocalStorage';
import { Task } from '../types/task';

/**
 * Hook to manage feature-specific state (tags, priorities, comments, etc.)
 */
export function useKanbanFeatures(tasks: Task[]) {
  // Tags
  const [taskTags, setTaskTags] = useState<Map<string, string[]>>(new Map());
  const [customLabels, setCustomLabels] = useState<Map<string, string>>(new Map());
  
  // Priorities
  const [taskPriorities, setTaskPriorities] = useState<Map<string, 'low' | 'medium' | 'high' | 'urgent'>>(new Map());
  
  // Comments
  const [taskComments, setTaskComments] = useState<Map<string, Array<{id: string; text: string; timestamp: Date; author: string}>>>(new Map());
  
  // Reminders
  const [reminders, setReminders] = useState<Array<{id: string; taskId: string; message: string; time: Date}>>([]);
  
  // Milestones
  const [milestones, setMilestones] = useState<Array<{id: string; name: string; date: Date; tasks: string[]}>>([]);
  
  // Dependencies
  const [taskDependencies, setTaskDependencies] = useState<Map<string, string[]>>(new Map());
  
  // Task history
  const [taskHistory, setTaskHistory] = useState<Map<string, Array<{timestamp: Date; changes: Partial<Task>}>>>(new Map());
  
  // Activity feed
  const [activityFeed, setActivityFeed] = useState<Array<{id: string; type: string; message: string; timestamp: Date; taskId?: string}>>([]);
  
  // Smart alerts
  const [smartAlerts, setSmartAlerts] = useState<Array<{id: string; type: string; message: string; timestamp: Date}>>([]);
  
  // Audit logs
  const [auditLogs, setAuditLogs] = useState<Array<{id: string; action: string; details: string; timestamp: Date}>>([]);
  
  // Templates
  const [taskTemplates, setTaskTemplates] = useLocalStorage<Array<{id: string; name: string; instruction: string; model?: string}>>('kanban-templates', []);
  
  // Custom views
  const [customViews, setCustomViews] = useLocalStorage<Array<{id: string; name: string; filters: any}>>('kanban-custom-views', []);
  const [activeCustomView, setActiveCustomView] = useState<string | null>(null);
  
  // Custom columns
  const [customColumns, setCustomColumns] = useLocalStorage<string[]>('kanban-custom-columns', []);
  
  // Dashboard layout
  const [dashboardLayout, setDashboardLayout] = useLocalStorage<'grid' | 'list' | 'compact'>('kanban-dashboard-layout', 'grid');
  
  // Auto reports
  const [autoReports, setAutoReports] = useState(false);
  const [reportFrequency, setReportFrequency] = useLocalStorage<'daily' | 'weekly' | 'monthly'>('kanban-report-frequency', 'daily');
  
  // Sound
  const [soundEnabled, setSoundEnabled] = useLocalStorage<boolean>('kanban-sound-enabled', false);
  
  // Quick actions
  const [quickActions, setQuickActions] = useState<Array<{id: string; name: string; action: () => void}>>([]);
  
  // Bulk operations
  const [bulkOperations, setBulkOperations] = useState<Array<string>>([]);

  // Auto-extract tags from tasks
  useEffect(() => {
    const newTags = new Map<string, string[]>();
    tasks.forEach(task => {
      const tags: string[] = [];
      const instruction = task.instruction?.toLowerCase() || '';
      
      if (instruction.includes('urgent') || instruction.includes('urgente')) tags.push('urgent');
      if (instruction.includes('bug') || instruction.includes('error')) tags.push('bug');
      if (instruction.includes('feature') || instruction.includes('nueva funcionalidad')) tags.push('feature');
      if (instruction.includes('refactor')) tags.push('refactor');
      if (instruction.includes('test') || instruction.includes('prueba')) tags.push('test');
      if (instruction.includes('documentation') || instruction.includes('documentación')) tags.push('documentation');
      
      if (tags.length > 0) {
        newTags.set(task.id, tags);
      }
    });
    setTaskTags(newTags);
  }, [tasks]);

  // Auto-detect priorities
  useEffect(() => {
    const newPriorities = new Map<string, 'low' | 'medium' | 'high' | 'urgent'>();
    tasks.forEach(task => {
      const instruction = task.instruction?.toLowerCase() || '';
      const error = task.error?.toLowerCase() || '';
      
      if (instruction.includes('urgent') || error.includes('critical')) {
        newPriorities.set(task.id, 'urgent');
      } else if (instruction.includes('important') || task.status === 'failed') {
        newPriorities.set(task.id, 'high');
      } else if (instruction.includes('priority') || task.status === 'processing') {
        newPriorities.set(task.id, 'medium');
      } else {
        newPriorities.set(task.id, 'low');
      }
    });
    setTaskPriorities(newPriorities);
  }, [tasks]);

  // Auto-detect milestones
  useEffect(() => {
    const newMilestones: Array<{id: string; name: string; date: Date; tasks: string[]}> = [];
    const milestoneKeywords = ['milestone', 'hito', 'release', 'sprint'];
    
    tasks.forEach(task => {
      const instruction = task.instruction?.toLowerCase() || '';
      milestoneKeywords.forEach(keyword => {
        if (instruction.includes(keyword)) {
          const existingMilestone = newMilestones.find(m => m.name.toLowerCase().includes(keyword));
          if (existingMilestone) {
            existingMilestone.tasks.push(task.id);
          } else {
            newMilestones.push({
              id: `milestone-${newMilestones.length}`,
              name: `Milestone: ${keyword}`,
              date: new Date(task.createdAt),
              tasks: [task.id]
            });
          }
        }
      });
    });
    
    setMilestones(newMilestones);
  }, [tasks]);

  return {
    // Tags
    taskTags,
    setTaskTags,
    customLabels,
    setCustomLabels,
    
    // Priorities
    taskPriorities,
    setTaskPriorities,
    
    // Comments
    taskComments,
    setTaskComments,
    
    // Reminders
    reminders,
    setReminders,
    
    // Milestones
    milestones,
    setMilestones,
    
    // Dependencies
    taskDependencies,
    setTaskDependencies,
    
    // History
    taskHistory,
    setTaskHistory,
    
    // Activity
    activityFeed,
    setActivityFeed,
    
    // Alerts
    smartAlerts,
    setSmartAlerts,
    
    // Audit
    auditLogs,
    setAuditLogs,
    
    // Templates
    taskTemplates,
    setTaskTemplates,
    
    // Views
    customViews,
    setCustomViews,
    activeCustomView,
    setActiveCustomView,
    
    // Columns
    customColumns,
    setCustomColumns,
    
    // Dashboard
    dashboardLayout,
    setDashboardLayout,
    
    // Reports
    autoReports,
    setAutoReports,
    reportFrequency,
    setReportFrequency,
    
    // Sound
    soundEnabled,
    setSoundEnabled,
    
    // Actions
    quickActions,
    setQuickActions,
    bulkOperations,
    setBulkOperations,
  };
}

