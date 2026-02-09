'use client';

import React, { useState, Suspense, lazy } from 'react';
import dynamic from 'next/dynamic';
import { useTabs } from '@/lib/hooks/useTabs';

// Core components - always loaded
import RobotControl from './RobotControl';
import StatusPanel from './StatusPanel';
import MetricsPanel from './MetricsPanel';
import SearchBar from './SearchBar';
import QuickStats from './QuickStats';
import QuickActions from './QuickActions';
import LanguageSelector from './LanguageSelector';
import KeyboardNavigation from './KeyboardNavigation';
import OnboardingTour from './OnboardingTour';
import ToastContainer from './ToastContainer';

// Lazy load heavy components for better performance
const ChatPanel = lazy(() => import('./ChatPanel'));
const Fullscreen3D = lazy(() => import('./Fullscreen3D'));
const MovementHistory = lazy(() => import('./MovementHistory'));
const TrajectoryOptimizer = lazy(() => import('./TrajectoryOptimizer'));
const RecordingPanel = lazy(() => import('./RecordingPanel'));
const SettingsPanel = lazy(() => import('./SettingsPanel'));
const LogsPanel = lazy(() => import('./LogsPanel'));
const AlertsPanel = lazy(() => import('./AlertsPanel'));
const AdvancedMetrics = lazy(() => import('./AdvancedMetrics'));
const TrajectoryComparison = lazy(() => import('./TrajectoryComparison'));
const CustomCommands = lazy(() => import('./CustomCommands'));
const HelpPanel = lazy(() => import('./HelpPanel'));
const WidgetDashboard = lazy(() => import('./WidgetDashboard'));
const ReportsPanel = lazy(() => import('./ReportsPanel'));
const PredictiveAnalysis = lazy(() => import('./PredictiveAnalysis'));
const CollaborationPanel = lazy(() => import('./CollaborationPanel'));
const AuthPanel = lazy(() => import('./AuthPanel'));
const PerformanceMonitor = lazy(() => import('./PerformanceMonitor'));
const BackendIntegrations = lazy(() => import('./BackendIntegrations'));
const DataExport = lazy(() => import('./DataExport'));
const SystemDiagnostics = lazy(() => import('./SystemDiagnostics'));
const RealTimeVisualization = lazy(() => import('./RealTimeVisualization'));
const EnergyOptimization = lazy(() => import('./EnergyOptimization'));
const SafetyMonitor = lazy(() => import('./SafetyMonitor'));
const CommandHistory = lazy(() => import('./CommandHistory'));
const PresetPositions = lazy(() => import('./PresetPositions'));
const SystemBackup = lazy(() => import('./SystemBackup'));
const NotificationCenter = lazy(() => import('./NotificationCenter'));
const ShortcutsGuide = lazy(() => import('./ShortcutsGuide'));
const SystemInfo = lazy(() => import('./SystemInfo'));
const ActivityTimeline = lazy(() => import('./ActivityTimeline'));
const ThemeCustomizer = lazy(() => import('./ThemeCustomizer'));
const DataVisualization = lazy(() => import('./DataVisualization'));
const RemoteControl = lazy(() => import('./RemoteControl'));
const CalibrationPanel = lazy(() => import('./CalibrationPanel'));
const MaintenancePanel = lazy(() => import('./MaintenancePanel'));
const DocumentationViewer = lazy(() => import('./DocumentationViewer'));
const LicenseManager = lazy(() => import('./LicenseManager'));
const UpdateChecker = lazy(() => import('./UpdateChecker'));
const FeedbackPanel = lazy(() => import('./FeedbackPanel'));
const AboutPanel = lazy(() => import('./AboutPanel'));
const FavoritesManager = lazy(() => import('./FavoritesManager'));
const UsageStatistics = lazy(() => import('./UsageStatistics'));
const AccessibilityPanel = lazy(() => import('./AccessibilityPanel'));
const PerformanceOptimizer = lazy(() => import('./PerformanceOptimizer'));
const TemplatesManager = lazy(() => import('./TemplatesManager'));
const AdvancedSearch = lazy(() => import('./AdvancedSearch'));
const PluginManager = lazy(() => import('./PluginManager'));
const WorkspaceManager = lazy(() => import('./WorkspaceManager'));
const SessionRecorder = lazy(() => import('./SessionRecorder'));
const QuickAccess = lazy(() => import('./QuickAccess'));
const DataAnalytics = lazy(() => import('./DataAnalytics'));
const AutomationPanel = lazy(() => import('./AutomationPanel'));
const CloudSync = lazy(() => import('./CloudSync'));
const AIAssistant = lazy(() => import('./AIAssistant'));
const LearningCenter = lazy(() => import('./LearningCenter'));
const Marketplace = lazy(() => import('./Marketplace'));
const VoiceControl = lazy(() => import('./VoiceControl'));
const GestureControl = lazy(() => import('./GestureControl'));
const ARView = lazy(() => import('./ARView'));
const TrajectorySimulator = lazy(() => import('./TrajectorySimulator'));
const CodeEditor = lazy(() => import('./CodeEditor'));
const Terminal = lazy(() => import('./Terminal'));
const VersionControl = lazy(() => import('./VersionControl'));
const TestSuite = lazy(() => import('./TestSuite'));
const PermissionsManager = lazy(() => import('./PermissionsManager'));
const APIIntegrations = lazy(() => import('./APIIntegrations'));
const BackupScheduler = lazy(() => import('./BackupScheduler'));
const SmartAlerts = lazy(() => import('./SmartAlerts'));
const RealTimeDashboard = lazy(() => import('./RealTimeDashboard'));
const EventLog = lazy(() => import('./EventLog'));
const SystemHealth = lazy(() => import('./SystemHealth'));
const NotificationSettings = lazy(() => import('./NotificationSettings'));
const DataSync = lazy(() => import('./DataSync'));
const ResourceMonitor = lazy(() => import('./ResourceMonitor'));
const ActivityFeed = lazy(() => import('./ActivityFeed'));
const AuditLog = lazy(() => import('./AuditLog'));
const UserManagement = lazy(() => import('./UserManagement'));
const ReportGenerator = lazy(() => import('./ReportGenerator'));
const ServiceStatus = lazy(() => import('./ServiceStatus'));
const AdvancedAnalytics = lazy(() => import('./AdvancedAnalytics'));
const ConfigurationManager = lazy(() => import('./ConfigurationManager'));
const DataBackup = lazy(() => import('./DataBackup'));
const NetworkMonitor = lazy(() => import('./NetworkMonitor'));
const SecurityCenter = lazy(() => import('./SecurityCenter'));
const PerformanceTuning = lazy(() => import('./PerformanceTuning'));
const ErrorTracker = lazy(() => import('./ErrorTracker'));
const SystemMetrics = lazy(() => import('./SystemMetrics'));
const TaskScheduler = lazy(() => import('./TaskScheduler'));
const DataExplorer = lazy(() => import('./DataExplorer'));
const LogAnalyzer = lazy(() => import('./LogAnalyzer'));
const APIDocumentation = lazy(() => import('./APIDocumentation'));
const QuickActionsPanel = lazy(() => import('./QuickActionsPanel'));
const CommandPalette = lazy(() => import('./CommandPalette'));
const WorkflowBuilder = lazy(() => import('./WorkflowBuilder'));
const DataPipeline = lazy(() => import('./DataPipeline'));
const MonitoringDashboard = lazy(() => import('./MonitoringDashboard'));
const SettingsAdvanced = lazy(() => import('./SettingsAdvanced'));
const AlertManager = lazy(() => import('./AlertManager'));
const PerformanceProfiler = lazy(() => import('./PerformanceProfiler'));
const SystemTweaks = lazy(() => import('./SystemTweaks'));

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center p-8">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-tesla-blue"></div>
  </div>
);
import { useRobotStore } from '@/lib/store/robotStore';
import { Activity, MessageSquare, Settings, BarChart3, Box, History, Sparkles, CircleDot, Cog, Terminal as TerminalIcon, Bell, TrendingUp, GitCompare, Command, HelpCircle, LayoutDashboard, FileText, Brain, Users, Lock, Gauge, Plug, Database, Stethoscope, Radio, Battery, Shield, Clock, MapPin, HardDrive, Keyboard, Info, Timeline, Palette, BarChart, Gamepad2, Target, Wrench, Book, Key, RefreshCw, MessageSquare as MessageSquareIcon, Star, BarChart as BarChartIcon, Accessibility, Zap, File, Search as SearchIcon, Puzzle, FolderOpen, Video, Zap as ZapIcon, TrendingUp as TrendingUpIcon, Bot, Cloud, Sparkles as SparklesIcon, GraduationCap, Store, Mic, Hand, Camera, Play, Code, GitBranch, TestTube, Shield as ShieldIcon, Calendar, Bell as BellIcon, Activity as ActivityIcon, FileText as FileTextIcon, Heart, Bell as BellSettingsIcon, RefreshCw as RefreshCwIcon, Monitor, Activity as ActivityFeedIcon, Shield as AuditIcon, Users as UsersIcon, FileText as ReportIcon, Server, TrendingUp as AnalyticsIcon, Settings as ConfigIcon, HardDrive as BackupIcon, Network, Shield as SecurityIcon, Gauge as TuningIcon, Bug, BarChart3 as MetricsIcon, Calendar as TaskIcon, Database as DataIcon, FileSearch, Stethoscope as DiagIcon, Book as BookIcon, Zap as ZapPanelIcon, Info as InfoIcon, Command as CommandIcon, Workflow, GitBranch as PipelineIcon, Monitor as MonitorIcon, Settings as SettingsAdvancedIcon, BarChart3 as DataVizIcon, Bell as AlertMgrIcon, Gauge as ProfilerIcon, Sliders, ChevronDown, X } from 'lucide-react';

type Tab = 'control' | 'chat' | 'status' | 'metrics' | '3d' | 'history' | 'optimize' | 'recording' | 'settings' | 'logs' | 'alerts' | 'advanced' | 'compare' | 'commands' | 'help' | 'widgets' | 'reports' | 'predictive' | 'collaboration' | 'auth' | 'performance' | 'integrations' | 'export' | 'diagnostics' | 'realtime' | 'energy' | 'safety' | 'cmdhistory' | 'presets' | 'backup' | 'notifications' | 'shortcuts' | 'systeminfo' | 'timeline' | 'theme' | 'visualization' | 'remote' | 'calibration' | 'maintenance' | 'docs' | 'license' | 'updates' | 'feedback' | 'about' | 'favorites' | 'usage' | 'accessibility' | 'optimizer' | 'templates' | 'advanced-search' | 'plugins' | 'workspaces' | 'sessions' | 'quick-access' | 'analytics' | 'automation' | 'cloud' | 'ai-assistant' | 'learning' | 'marketplace' | 'voice' | 'gestures' | 'ar' | 'simulator' | 'code' | 'terminal' | 'versions' | 'tests' | 'permissions' | 'api-integrations' | 'backup-scheduler' | 'smart-alerts' | 'realtime-dashboard' | 'event-log' | 'system-health' | 'notification-settings' | 'data-sync' | 'resource-monitor' | 'activity-feed' | 'audit-log' | 'user-management' | 'report-generator' | 'service-status' | 'advanced-analytics' | 'configuration-manager' | 'data-backup' | 'network-monitor' | 'security-center' | 'performance-tuning' | 'error-tracker' | 'system-metrics' | 'task-scheduler' | 'data-explorer' | 'log-analyzer' | 'system-diagnostics' | 'api-docs' | 'quick-actions-panel' | 'system-info' | 'command-palette' | 'workflow-builder' | 'data-pipeline' | 'monitoring-dashboard' | 'settings-advanced' | 'data-visualization' | 'alert-manager' | 'performance-profiler' | 'system-tweaks';

// Tab Dropdown Menu Component
function TabDropdownMenu({ 
  tabs, 
  activeTab, 
  onTabSelect 
}: { 
  tabs: { id: Tab; label: string; icon: React.ReactNode }[]; 
  activeTab: Tab; 
  onTabSelect: (tab: Tab) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const filteredTabs = tabs.filter(tab =>
    tab.label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="relative flex-shrink-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-tesla-sm px-tesla-md py-tesla-sm text-sm font-medium transition-all whitespace-nowrap border-b-2 min-h-[48px] ${
          tabs.some(t => t.id === activeTab)
            ? 'text-tesla-blue border-tesla-blue bg-blue-50/50'
            : 'text-tesla-gray-dark border-transparent hover:text-tesla-black hover:border-gray-300 hover:bg-gray-50'
        }`}
      >
        <span>Más</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full left-0 mt-1 w-80 max-h-[600px] bg-white border border-gray-200 rounded-lg shadow-xl z-50 overflow-hidden">
            <div className="p-tesla-sm border-b border-gray-200 sticky top-0 bg-white">
              <div className="relative">
                <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Buscar pestaña..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-tesla-md py-tesla-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-tesla-blue"
                  autoFocus
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
            <div className="overflow-y-auto max-h-[540px]">
              {filteredTabs.length > 0 ? (
                filteredTabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => {
                      onTabSelect(tab.id);
                      setIsOpen(false);
                      setSearchQuery('');
                    }}
                    className={`w-full flex items-center gap-tesla-sm px-tesla-md py-tesla-sm text-sm transition-colors hover:bg-gray-50 ${
                      activeTab === tab.id ? 'bg-blue-50 text-tesla-blue' : 'text-tesla-gray-dark'
                    }`}
                  >
                    <span className="text-tesla-gray-dark">{tab.icon}</span>
                    <span className="flex-1 text-left">{tab.label}</span>
                    {activeTab === tab.id && (
                      <div className="w-2 h-2 rounded-full bg-tesla-blue" />
                    )}
                  </button>
                ))
              ) : (
                <div className="px-tesla-md py-tesla-xl text-center text-gray-500">
                  No se encontraron pestañas
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function Dashboard() {
  const { status, error } = useRobotStore();

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'control', label: 'Control', icon: <Activity className="w-5 h-5" /> },
    { id: 'chat', label: 'Chat', icon: <MessageSquare className="w-5 h-5" /> },
    { id: '3d', label: '3D View', icon: <Box className="w-5 h-5" /> },
    { id: 'status', label: 'Estado', icon: <Settings className="w-5 h-5" /> },
    { id: 'metrics', label: 'Métricas', icon: <BarChart3 className="w-5 h-5" /> },
    { id: 'history', label: 'Historial', icon: <History className="w-5 h-5" /> },
    { id: 'optimize', label: 'Optimizar', icon: <Sparkles className="w-5 h-5" /> },
    { id: 'recording', label: 'Grabación', icon: <CircleDot className="w-5 h-5" /> },
    { id: 'advanced', label: 'Métricas Avanzadas', icon: <TrendingUp className="w-5 h-5" /> },
    { id: 'compare', label: 'Comparar', icon: <GitCompare className="w-5 h-5" /> },
    { id: 'commands', label: 'Comandos', icon: <Command className="w-5 h-5" /> },
    { id: 'widgets', label: 'Widgets', icon: <LayoutDashboard className="w-5 h-5" /> },
    { id: 'reports', label: 'Reportes', icon: <FileText className="w-5 h-5" /> },
    { id: 'predictive', label: 'Predictivo', icon: <Brain className="w-5 h-5" /> },
    { id: 'collaboration', label: 'Colaboración', icon: <Users className="w-5 h-5" /> },
    { id: 'auth', label: 'Autenticación', icon: <Lock className="w-5 h-5" /> },
    { id: 'performance', label: 'Rendimiento', icon: <Gauge className="w-5 h-5" /> },
    { id: 'integrations', label: 'Integraciones', icon: <Plug className="w-5 h-5" /> },
    { id: 'export', label: 'Exportar', icon: <Database className="w-5 h-5" /> },
    { id: 'diagnostics', label: 'Diagnóstico', icon: <Stethoscope className="w-5 h-5" /> },
    { id: 'realtime', label: 'Tiempo Real', icon: <Radio className="w-5 h-5" /> },
    { id: 'energy', label: 'Energía', icon: <Battery className="w-5 h-5" /> },
    { id: 'safety', label: 'Seguridad', icon: <Shield className="w-5 h-5" /> },
    { id: 'cmdhistory', label: 'Hist. Comandos', icon: <Clock className="w-5 h-5" /> },
    { id: 'presets', label: 'Presets', icon: <MapPin className="w-5 h-5" /> },
    { id: 'backup', label: 'Backup', icon: <HardDrive className="w-5 h-5" /> },
    { id: 'notifications', label: 'Notificaciones', icon: <Bell className="w-5 h-5" /> },
    { id: 'shortcuts', label: 'Atajos', icon: <Keyboard className="w-5 h-5" /> },
    { id: 'systeminfo', label: 'Sistema', icon: <Info className="w-5 h-5" /> },
    { id: 'timeline', label: 'Línea de Tiempo', icon: <History className="w-5 h-5" /> },
    { id: 'theme', label: 'Tema', icon: <Palette className="w-5 h-5" /> },
    { id: 'visualization', label: 'Visualización', icon: <BarChart className="w-5 h-5" /> },
    { id: 'remote', label: 'Control Remoto', icon: <Gamepad2 className="w-5 h-5" /> },
    { id: 'calibration', label: 'Calibración', icon: <Target className="w-5 h-5" /> },
    { id: 'maintenance', label: 'Mantenimiento', icon: <Wrench className="w-5 h-5" /> },
    { id: 'docs', label: 'Documentación', icon: <Book className="w-5 h-5" /> },
    { id: 'license', label: 'Licencias', icon: <Key className="w-5 h-5" /> },
    { id: 'updates', label: 'Actualizaciones', icon: <RefreshCw className="w-5 h-5" /> },
    { id: 'feedback', label: 'Feedback', icon: <MessageSquareIcon className="w-5 h-5" /> },
    { id: 'about', label: 'Acerca de', icon: <Info className="w-5 h-5" /> },
    { id: 'favorites', label: 'Favoritos', icon: <Star className="w-5 h-5" /> },
    { id: 'usage', label: 'Uso', icon: <BarChartIcon className="w-5 h-5" /> },
    { id: 'accessibility', label: 'Accesibilidad', icon: <Accessibility className="w-5 h-5" /> },
    { id: 'optimizer', label: 'Optimizador', icon: <Zap className="w-5 h-5" /> },
    { id: 'templates', label: 'Plantillas', icon: <File className="w-5 h-5" /> },
    { id: 'advanced-search', label: 'Búsqueda Avanzada', icon: <SearchIcon className="w-5 h-5" /> },
    { id: 'plugins', label: 'Plugins', icon: <Puzzle className="w-5 h-5" /> },
    { id: 'workspaces', label: 'Workspaces', icon: <FolderOpen className="w-5 h-5" /> },
    { id: 'sessions', label: 'Sesiones', icon: <Video className="w-5 h-5" /> },
    { id: 'quick-access', label: 'Acceso Rápido', icon: <ZapIcon className="w-5 h-5" /> },
    { id: 'analytics', label: 'Análisis', icon: <TrendingUpIcon className="w-5 h-5" /> },
    { id: 'automation', label: 'Automatización', icon: <Bot className="w-5 h-5" /> },
    { id: 'cloud', label: 'Nube', icon: <Cloud className="w-5 h-5" /> },
    { id: 'ai-assistant', label: 'IA Asistente', icon: <SparklesIcon className="w-5 h-5" /> },
    { id: 'learning', label: 'Aprendizaje', icon: <GraduationCap className="w-5 h-5" /> },
    { id: 'marketplace', label: 'Marketplace', icon: <Store className="w-5 h-5" /> },
    { id: 'voice', label: 'Voz', icon: <Mic className="w-5 h-5" /> },
    { id: 'gestures', label: 'Gestos', icon: <Hand className="w-5 h-5" /> },
    { id: 'ar', label: 'AR', icon: <Camera className="w-5 h-5" /> },
    { id: 'simulator', label: 'Simulador', icon: <Play className="w-5 h-5" /> },
    { id: 'code', label: 'Editor', icon: <Code className="w-5 h-5" /> },
    { id: 'terminal', label: 'Terminal', icon: <TerminalIcon className="w-5 h-5" /> },
    { id: 'versions', label: 'Versiones', icon: <GitBranch className="w-5 h-5" /> },
    { id: 'tests', label: 'Pruebas', icon: <TestTube className="w-5 h-5" /> },
    { id: 'permissions', label: 'Permisos', icon: <ShieldIcon className="w-5 h-5" /> },
    { id: 'api-integrations', label: 'APIs', icon: <Plug className="w-5 h-5" /> },
    { id: 'backup-scheduler', label: 'Backups', icon: <Calendar className="w-5 h-5" /> },
    { id: 'smart-alerts', label: 'Alertas Inteligentes', icon: <BellIcon className="w-5 h-5" /> },
    { id: 'realtime-dashboard', label: 'Dashboard RT', icon: <ActivityIcon className="w-5 h-5" /> },
    { id: 'event-log', label: 'Eventos', icon: <FileTextIcon className="w-5 h-5" /> },
    { id: 'system-health', label: 'Salud', icon: <Heart className="w-5 h-5" /> },
    { id: 'notification-settings', label: 'Notificaciones', icon: <BellSettingsIcon className="w-5 h-5" /> },
    { id: 'data-sync', label: 'Sincronización', icon: <RefreshCwIcon className="w-5 h-5" /> },
    { id: 'resource-monitor', label: 'Recursos', icon: <Monitor className="w-5 h-5" /> },
    { id: 'activity-feed', label: 'Actividad', icon: <ActivityFeedIcon className="w-5 h-5" /> },
    { id: 'audit-log', label: 'Auditoría', icon: <AuditIcon className="w-5 h-5" /> },
    { id: 'user-management', label: 'Usuarios', icon: <UsersIcon className="w-5 h-5" /> },
    { id: 'report-generator', label: 'Reportes', icon: <ReportIcon className="w-5 h-5" /> },
    { id: 'service-status', label: 'Servicios', icon: <Server className="w-5 h-5" /> },
    { id: 'advanced-analytics', label: 'Análisis Avanzado', icon: <AnalyticsIcon className="w-5 h-5" /> },
    { id: 'configuration-manager', label: 'Configuración', icon: <ConfigIcon className="w-5 h-5" /> },
    { id: 'data-backup', label: 'Respaldo', icon: <BackupIcon className="w-5 h-5" /> },
    { id: 'network-monitor', label: 'Red', icon: <Network className="w-5 h-5" /> },
    { id: 'security-center', label: 'Seguridad', icon: <SecurityIcon className="w-5 h-5" /> },
    { id: 'performance-tuning', label: 'Ajuste', icon: <TuningIcon className="w-5 h-5" /> },
    { id: 'error-tracker', label: 'Errores', icon: <Bug className="w-5 h-5" /> },
    { id: 'system-metrics', label: 'Métricas', icon: <MetricsIcon className="w-5 h-5" /> },
    { id: 'task-scheduler', label: 'Tareas', icon: <TaskIcon className="w-5 h-5" /> },
    { id: 'data-explorer', label: 'Datos', icon: <DataIcon className="w-5 h-5" /> },
    { id: 'log-analyzer', label: 'Análisis Logs', icon: <FileSearch className="w-5 h-5" /> },
    { id: 'system-diagnostics', label: 'Diagnósticos', icon: <DiagIcon className="w-5 h-5" /> },
    { id: 'api-docs', label: 'API Docs', icon: <BookIcon className="w-5 h-5" /> },
    { id: 'quick-actions-panel', label: 'Acciones', icon: <ZapPanelIcon className="w-5 h-5" /> },
    { id: 'system-info', label: 'Info Sistema', icon: <InfoIcon className="w-5 h-5" /> },
    { id: 'command-palette', label: 'Paleta', icon: <CommandIcon className="w-5 h-5" /> },
    { id: 'workflow-builder', label: 'Workflows', icon: <Workflow className="w-5 h-5" /> },
    { id: 'data-pipeline', label: 'Pipeline', icon: <PipelineIcon className="w-5 h-5" /> },
    { id: 'monitoring-dashboard', label: 'Monitoreo', icon: <MonitorIcon className="w-5 h-5" /> },
    { id: 'settings-advanced', label: 'Config Avanzada', icon: <SettingsAdvancedIcon className="w-5 h-5" /> },
    { id: 'data-visualization', label: 'Visualización', icon: <DataVizIcon className="w-5 h-5" /> },
    { id: 'alert-manager', label: 'Gestor Alertas', icon: <AlertMgrIcon className="w-5 h-5" /> },
    { id: 'performance-profiler', label: 'Perfilador', icon: <ProfilerIcon className="w-5 h-5" /> },
    { id: 'system-tweaks', label: 'Ajustes', icon: <Sliders className="w-5 h-5" /> },
    { id: 'logs', label: 'Logs', icon: <TerminalIcon className="w-5 h-5" /> },
    { id: 'alerts', label: 'Alertas', icon: <Bell className="w-5 h-5" /> },
    { id: 'help', label: 'Ayuda', icon: <HelpCircle className="w-5 h-5" /> },
    { id: 'settings', label: 'Config', icon: <Cog className="w-5 h-5" /> },
  ];

  return (
    <div className="min-h-screen bg-white p-tesla-md md:p-tesla-lg lg:p-tesla-xl relative">
      <KeyboardNavigation />
      <OnboardingTour />
      <ToastContainer />
      {/* Header */}
      <header className="mb-tesla-lg md:mb-tesla-xl">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-tesla-md">
          <div>
            <h1 className="text-3xl md:text-4xl font-semibold text-tesla-black mb-tesla-sm tracking-tight">
              Robot Movement AI
            </h1>
            <p className="text-tesla-gray-dark text-sm md:text-base">
              Plataforma IA de Movimiento Robótico
            </p>
          </div>
          <div className="flex items-center gap-tesla-sm md:gap-tesla-md flex-wrap">
            <SearchBar
              onResultSelect={(result) => {
                if (result.type === 'tab') {
                  const tabId = result.id.replace('tab-', '') as Tab;
                  setActiveTab(tabId);
                }
              }}
              tabs={tabs.map((tab) => ({
                id: tab.id,
                label: tab.label,
                action: () => setActiveTab(tab.id),
              }))}
            />
            <LanguageSelector />
            <div
              className={`px-tesla-sm md:px-tesla-md py-tesla-sm rounded-md text-xs md:text-sm font-medium ${
                status?.robot_status.connected
                  ? 'bg-green-50 text-green-700 border border-green-200'
                  : 'bg-red-50 text-red-700 border border-red-200'
              }`}
              role="status"
              aria-live="polite"
            >
              {status?.robot_status.connected ? 'Conectado' : 'Desconectado'}
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="mb-tesla-md p-tesla-md bg-red-50 border border-red-200 rounded-md text-red-700 flex items-center justify-between">
          <div className="flex items-center gap-tesla-sm">
            <span className="font-medium">Error de conexión:</span>
            <span>{error}</span>
          </div>
          <button
            onClick={() => {
              const store = useRobotStore.getState();
              store.clearError();
              store.fetchStatus();
            }}
            className="px-tesla-sm py-tesla-xs text-sm bg-red-100 hover:bg-red-200 rounded transition-colors"
          >
            Reintentar
          </button>
        </div>
      )}

      {/* Tabs - Improved Layout */}
      <div className="mb-tesla-lg relative">
        <div className="flex items-center gap-tesla-xs border-b border-gray-200 overflow-x-auto scrollbar-hide scroll-smooth" role="tablist" aria-label="Navegación de pestañas" style={{ scrollbarWidth: 'thin' }}>
          {/* Primary Tabs - Most Used (8 most common) */}
          {tabs.slice(0, 8).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              id={`tab-${tab.id}`}
              className={`flex items-center gap-tesla-sm px-tesla-md py-tesla-sm text-sm font-medium transition-all whitespace-nowrap border-b-2 min-h-[48px] min-w-[100px] flex-shrink-0 ${
                activeTab === tab.id
                  ? 'text-tesla-blue border-tesla-blue bg-blue-50/50 font-semibold'
                  : 'text-tesla-gray-dark border-transparent hover:text-tesla-black hover:border-gray-300 hover:bg-gray-50'
              }`}
            >
              <span className="flex-shrink-0">{tab.icon}</span>
              <span className="truncate">{tab.label}</span>
            </button>
          ))}
          
          {/* More Menu for Secondary Tabs */}
          <TabDropdownMenu 
            tabs={tabs.slice(8)} 
            activeTab={activeTab}
            onTabSelect={setActiveTab}
          />
        </div>
      </div>

      {/* Content */}
      <div className="p-tesla-md md:p-tesla-lg lg:p-tesla-xl">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-tesla-md md:gap-tesla-lg">
        <div className="lg:col-span-2" role="tabpanel" aria-labelledby={`tab-${activeTab}`} id={`panel-${activeTab}`}>
          <Suspense fallback={<LoadingFallback />}>
            {activeTab === 'control' && <RobotControl />}
            {activeTab === 'chat' && <ChatPanel />}
            {activeTab === '3d' && <Fullscreen3D />}
            {activeTab === 'status' && <StatusPanel />}
            {activeTab === 'metrics' && <MetricsPanel />}
            {activeTab === 'history' && <MovementHistory />}
            {activeTab === 'optimize' && <TrajectoryOptimizer />}
            {activeTab === 'recording' && <RecordingPanel />}
            {activeTab === 'advanced' && <AdvancedMetrics />}
            {activeTab === 'compare' && <TrajectoryComparison />}
            {activeTab === 'commands' && <CustomCommands />}
            {activeTab === 'widgets' && <WidgetDashboard />}
            {activeTab === 'reports' && <ReportsPanel />}
            {activeTab === 'predictive' && <PredictiveAnalysis />}
            {activeTab === 'collaboration' && <CollaborationPanel />}
            {activeTab === 'auth' && <AuthPanel />}
            {activeTab === 'performance' && <PerformanceMonitor />}
            {activeTab === 'integrations' && <BackendIntegrations />}
            {activeTab === 'export' && <DataExport />}
            {activeTab === 'diagnostics' && <SystemDiagnostics />}
            {activeTab === 'realtime' && <RealTimeVisualization />}
            {activeTab === 'energy' && <EnergyOptimization />}
            {activeTab === 'safety' && <SafetyMonitor />}
            {activeTab === 'cmdhistory' && <CommandHistory />}
            {activeTab === 'presets' && <PresetPositions />}
            {activeTab === 'backup' && <SystemBackup />}
            {activeTab === 'notifications' && <NotificationCenter />}
            {activeTab === 'shortcuts' && <ShortcutsGuide />}
            {activeTab === 'systeminfo' && <SystemInfo />}
            {activeTab === 'timeline' && <ActivityTimeline />}
            {activeTab === 'theme' && <ThemeCustomizer />}
            {activeTab === 'visualization' && <DataVisualization />}
            {activeTab === 'remote' && <RemoteControl />}
            {activeTab === 'calibration' && <CalibrationPanel />}
            {activeTab === 'maintenance' && <MaintenancePanel />}
            {activeTab === 'docs' && <DocumentationViewer />}
            {activeTab === 'license' && <LicenseManager />}
            {activeTab === 'updates' && <UpdateChecker />}
            {activeTab === 'feedback' && <FeedbackPanel />}
            {activeTab === 'about' && <AboutPanel />}
            {activeTab === 'favorites' && <FavoritesManager />}
            {activeTab === 'usage' && <UsageStatistics />}
            {activeTab === 'accessibility' && <AccessibilityPanel />}
            {activeTab === 'optimizer' && <PerformanceOptimizer />}
            {activeTab === 'templates' && <TemplatesManager />}
            {activeTab === 'advanced-search' && <AdvancedSearch />}
            {activeTab === 'plugins' && <PluginManager />}
            {activeTab === 'workspaces' && <WorkspaceManager />}
            {activeTab === 'sessions' && <SessionRecorder />}
            {activeTab === 'quick-access' && <QuickAccess />}
            {activeTab === 'analytics' && <DataAnalytics />}
            {activeTab === 'automation' && <AutomationPanel />}
            {activeTab === 'cloud' && <CloudSync />}
            {activeTab === 'ai-assistant' && <AIAssistant />}
            {activeTab === 'learning' && <LearningCenter />}
            {activeTab === 'marketplace' && <Marketplace />}
            {activeTab === 'voice' && <VoiceControl />}
            {activeTab === 'gestures' && <GestureControl />}
            {activeTab === 'ar' && <ARView />}
            {activeTab === 'simulator' && <TrajectorySimulator />}
            {activeTab === 'code' && <CodeEditor />}
            {activeTab === 'terminal' && <Terminal />}
            {activeTab === 'versions' && <VersionControl />}
            {activeTab === 'tests' && <TestSuite />}
            {activeTab === 'permissions' && <PermissionsManager />}
            {activeTab === 'api-integrations' && <APIIntegrations />}
            {activeTab === 'backup-scheduler' && <BackupScheduler />}
            {activeTab === 'smart-alerts' && <SmartAlerts />}
            {activeTab === 'realtime-dashboard' && <RealTimeDashboard />}
            {activeTab === 'event-log' && <EventLog />}
            {activeTab === 'system-health' && <SystemHealth />}
            {activeTab === 'notification-settings' && <NotificationSettings />}
            {activeTab === 'data-sync' && <DataSync />}
            {activeTab === 'resource-monitor' && <ResourceMonitor />}
            {activeTab === 'activity-feed' && <ActivityFeed />}
            {activeTab === 'audit-log' && <AuditLog />}
            {activeTab === 'user-management' && <UserManagement />}
            {activeTab === 'report-generator' && <ReportGenerator />}
            {activeTab === 'service-status' && <ServiceStatus />}
            {activeTab === 'advanced-analytics' && <AdvancedAnalytics />}
            {activeTab === 'configuration-manager' && <ConfigurationManager />}
            {activeTab === 'data-backup' && <DataBackup />}
            {activeTab === 'network-monitor' && <NetworkMonitor />}
            {activeTab === 'security-center' && <SecurityCenter />}
            {activeTab === 'performance-tuning' && <PerformanceTuning />}
            {activeTab === 'error-tracker' && <ErrorTracker />}
            {activeTab === 'system-metrics' && <SystemMetrics />}
            {activeTab === 'task-scheduler' && <TaskScheduler />}
            {activeTab === 'data-explorer' && <DataExplorer />}
            {activeTab === 'log-analyzer' && <LogAnalyzer />}
            {activeTab === 'system-diagnostics' && <SystemDiagnostics />}
            {activeTab === 'api-docs' && <APIDocumentation />}
            {activeTab === 'quick-actions-panel' && <QuickActionsPanel />}
            {activeTab === 'system-info' && <SystemInfo />}
            {activeTab === 'command-palette' && <CommandPalette />}
            {activeTab === 'workflow-builder' && <WorkflowBuilder />}
            {activeTab === 'data-pipeline' && <DataPipeline />}
            {activeTab === 'monitoring-dashboard' && <MonitoringDashboard />}
            {activeTab === 'settings-advanced' && <SettingsAdvanced />}
            {activeTab === 'data-visualization' && <DataVisualization />}
            {activeTab === 'alert-manager' && <AlertManager />}
            {activeTab === 'performance-profiler' && <PerformanceProfiler />}
            {activeTab === 'system-tweaks' && <SystemTweaks />}
            {activeTab === 'logs' && <LogsPanel />}
            {activeTab === 'alerts' && <AlertsPanel />}
            {activeTab === 'help' && <HelpPanel />}
            {activeTab === 'settings' && <SettingsPanel />}
          </Suspense>
        </div>
        <div className="lg:col-span-1 space-y-tesla-lg">
          <QuickStats />
          <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
            <h2 className="text-xl font-semibold text-tesla-black mb-tesla-md">
              Información Rápida
            </h2>
            <div className="space-y-tesla-md">
              <div>
                <p className="text-tesla-gray-dark text-sm mb-tesla-xs">Marca del Robot</p>
                <p className="text-tesla-black font-medium">
                  {status?.config.robot_brand || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-tesla-gray-dark text-sm mb-1">ROS Habilitado</p>
                <p className="text-tesla-black font-medium">
                  {status?.config.ros_enabled ? 'Sí' : 'No'}
                </p>
              </div>
              <div>
                <p className="text-tesla-gray-dark text-sm mb-1">Frecuencia de Feedback</p>
                <p className="text-tesla-black font-medium">
                  {status?.config.feedback_frequency || 0} Hz
                </p>
              </div>
              {status?.robot_status.position && (
                <div>
                  <p className="text-tesla-gray-dark text-sm mb-1">Posición Actual</p>
                  <p className="text-tesla-black font-medium text-sm">
                    X: {status.robot_status.position.x.toFixed(3)} | Y:{' '}
                    {status.robot_status.position.y.toFixed(3)} | Z:{' '}
                    {status.robot_status.position.z.toFixed(3)}
                  </p>
                </div>
              )}
            </div>
          </div>
          <QuickActions />
        </div>
        </div>
      </div>
    </div>
  );
}

