import { useState } from 'react';
import { useLocalStorage } from './useLocalStorage';

/**
 * Hook to manage all modal and panel states in the Kanban page
 */
export function useKanbanModals() {
  // Modal states
  const [showNotifications, setShowNotifications] = useState(false);
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);
  const [showTaskComparison, setShowTaskComparison] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [showTagManager, setShowTagManager] = useState(false);
  const [showShareMenu, setShowShareMenu] = useState(false);
  const [showNetworkView, setShowNetworkView] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showComments, setShowComments] = useState<Map<string, boolean>>(new Map());
  const [showReminders, setShowReminders] = useState(false);
  const [showMilestones, setShowMilestones] = useState(false);
  const [showDependencies, setShowDependencies] = useState(false);
  const [showCharts, setShowCharts] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [showProgressView, setShowProgressView] = useState(false);
  const [showExecutiveSummary, setShowExecutiveSummary] = useState(false);
  const [showCollaboration, setShowCollaboration] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const [showSmartAlerts, setShowSmartAlerts] = useState(false);
  const [showActivityFeed, setShowActivityFeed] = useState(false);
  const [showCustomDashboard, setShowCustomDashboard] = useState(false);
  const [showSavedFilters, setShowSavedFilters] = useState(false);
  const [showLabelManager, setShowLabelManager] = useState(false);
  const [showAuditLogs, setShowAuditLogs] = useState(false);
  const [showBackupRestore, setShowBackupRestore] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);
  const [showBulkMenu, setShowBulkMenu] = useState(false);
  const [showRepoSummary, setShowRepoSummary] = useState(false);
  const [showRecentActivity, setShowRecentActivity] = useState(false);
  const [showAnalytics, setShowAnalytics] = useLocalStorage<boolean>('kanban-show-analytics', false);
  
  // Presentation mode
  const [presentationMode, setPresentationMode] = useState(false);

  // Close all modals
  const closeAllModals = () => {
    setShowNotifications(false);
    setShowAdvancedSearch(false);
    setShowTaskComparison(false);
    setShowExportMenu(false);
    setShowTagManager(false);
    setShowShareMenu(false);
    setShowNetworkView(false);
    setShowHeatmap(false);
    setShowReminders(false);
    setShowMilestones(false);
    setShowDependencies(false);
    setShowCharts(false);
    setShowShareDialog(false);
    setShowProgressView(false);
    setShowExecutiveSummary(false);
    setShowCollaboration(false);
    setShowTemplates(false);
    setShowSmartAlerts(false);
    setShowActivityFeed(false);
    setShowCustomDashboard(false);
    setShowSavedFilters(false);
    setShowLabelManager(false);
    setShowAuditLogs(false);
    setShowBackupRestore(false);
    setShowComparison(false);
    setShowQuickActions(false);
    setShowBulkMenu(false);
    setShowRepoSummary(false);
    setShowRecentActivity(false);
  };

  return {
    // Modal states
    showNotifications,
    setShowNotifications,
    showAdvancedSearch,
    setShowAdvancedSearch,
    showTaskComparison,
    setShowTaskComparison,
    showExportMenu,
    setShowExportMenu,
    showTagManager,
    setShowTagManager,
    showShareMenu,
    setShowShareMenu,
    showNetworkView,
    setShowNetworkView,
    showHeatmap,
    setShowHeatmap,
    showComments,
    setShowComments,
    showReminders,
    setShowReminders,
    showMilestones,
    setShowMilestones,
    showDependencies,
    setShowDependencies,
    showCharts,
    setShowCharts,
    showShareDialog,
    setShowShareDialog,
    showProgressView,
    setShowProgressView,
    showExecutiveSummary,
    setShowExecutiveSummary,
    showCollaboration,
    setShowCollaboration,
    showTemplates,
    setShowTemplates,
    showSmartAlerts,
    setShowSmartAlerts,
    showActivityFeed,
    setShowActivityFeed,
    showCustomDashboard,
    setShowCustomDashboard,
    showSavedFilters,
    setShowSavedFilters,
    showLabelManager,
    setShowLabelManager,
    showAuditLogs,
    setShowAuditLogs,
    showBackupRestore,
    setShowBackupRestore,
    showComparison,
    setShowComparison,
    showQuickActions,
    setShowQuickActions,
    showBulkMenu,
    setShowBulkMenu,
    showRepoSummary,
    setShowRepoSummary,
    showRecentActivity,
    setShowRecentActivity,
    showAnalytics,
    setShowAnalytics,
    presentationMode,
    setPresentationMode,
    closeAllModals,
  };
}

