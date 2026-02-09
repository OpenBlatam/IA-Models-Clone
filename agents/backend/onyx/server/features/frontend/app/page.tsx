'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/app-store';
import { apiClient } from '@/lib/api-client';
import { useHotkeys } from 'react-hotkeys-hook';
import Header from '@/components/Header';
import Sidebar from '@/components/Sidebar';
import Dashboard from '@/components/Dashboard';
import GenerateView from '@/components/views/GenerateView';
import TasksView from '@/components/views/TasksView';
import DocumentsView from '@/components/views/DocumentsView';
import StatsView from '@/components/views/StatsView';
import FavoritesView from '@/components/views/FavoritesView';
import QuickActions from '@/components/QuickActions';
import GlobalSearch from '@/components/GlobalSearch';
import CommandPalette from '@/components/CommandPalette';
import KeyboardShortcuts from '@/components/KeyboardShortcuts';
import OfflineIndicator from '@/components/OfflineIndicator';
import PerformanceMonitor from '@/components/PerformanceMonitor';
import SyncStatus from '@/components/SyncStatus';
import QuickNotes from '@/components/QuickNotes';
import BackupManager from '@/components/BackupManager';
import Tutorial from '@/components/Tutorial';
import ContextualHelp from '@/components/ContextualHelp';
import UpdateChecker from '@/components/UpdateChecker';
import WelcomeScreen from '@/components/WelcomeScreen';
import KeyboardNavigation from '@/components/KeyboardNavigation';
import DevTools from '@/components/DevTools';
import NetworkMonitor from '@/components/NetworkMonitor';
import ToastContainer from '@/components/ToastContainer';
import SmartSuggestions from '@/components/SmartSuggestions';
import NotificationSounds from '@/components/NotificationSounds';
import PerformanceOptimizer from '@/components/PerformanceOptimizer';

export default function Home() {
  const { activeView, setConnected, setHealth, setStats, setActiveView } = useAppStore();
  const [showTutorial, setShowTutorial] = useState(false);
  const [showWelcome, setShowWelcome] = useState(false);
  const [contextualHelpEnabled, setContextualHelpEnabled] = useState(false);

  // Check if tutorial was completed
  useEffect(() => {
    const completed = localStorage.getItem('bul_tutorial_completed');
    const welcomeShown = localStorage.getItem('bul_welcome_shown');
    
    if (!welcomeShown) {
      setTimeout(() => setShowWelcome(true), 500);
    } else if (!completed) {
      setTimeout(() => setShowTutorial(true), 1000);
    }
    
    // Check contextual help preference
    const helpEnabled = localStorage.getItem('bul_contextual_help') === 'true';
    setContextualHelpEnabled(helpEnabled);
  }, []);

  // Global keyboard shortcuts
  useHotkeys('ctrl+d,cmd+d', () => setActiveView('dashboard'), { preventDefault: true });
  useHotkeys('ctrl+g,cmd+g', () => setActiveView('generate'), { preventDefault: true });
  useHotkeys('ctrl+t,cmd+t', () => setActiveView('tasks'), { preventDefault: true });
  useHotkeys('ctrl+f,cmd+f', () => setActiveView('favorites'), { preventDefault: true });

  useEffect(() => {
    // Check health on mount
    const checkHealth = async () => {
      try {
        const health = await apiClient.getHealth();
        setHealth(health);
        setConnected(true);
      } catch (error) {
        console.error('Health check failed:', error);
        setConnected(false);
      }
    };

    checkHealth();
    const healthInterval = setInterval(checkHealth, 30000); // Check every 30s

    return () => clearInterval(healthInterval);
  }, [setHealth, setConnected]);

  useEffect(() => {
    // Load stats periodically
    const loadStats = async () => {
      try {
        const stats = await apiClient.getStats();
        setStats(stats);
      } catch (error) {
        console.error('Failed to load stats:', error);
      }
    };

    loadStats();
    const statsInterval = setInterval(loadStats, 10000); // Update every 10s

    return () => clearInterval(statsInterval);
  }, [setStats]);

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard />;
      case 'generate':
        return <GenerateView />;
      case 'tasks':
        return <TasksView />;
      case 'documents':
        return <DocumentsView />;
      case 'stats':
        return <StatsView />;
      case 'favorites':
        return <FavoritesView />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <OfflineIndicator />
      <Header />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6">{renderView()}</main>
      </div>
      <QuickActions
        onGenerate={() => setActiveView('generate')}
        onViewDocuments={() => setActiveView('documents')}
        onViewFavorites={() => setActiveView('favorites')}
      />
      <GlobalSearch />
      <CommandPalette />
      <KeyboardShortcuts />
      <PerformanceMonitor />
      <SyncStatus />
      <QuickNotes />
      <BackupManager />
      <Tutorial isOpen={showTutorial} onClose={() => setShowTutorial(false)} />
      <WelcomeScreen
        isOpen={showWelcome}
        onClose={() => {
          setShowWelcome(false);
          localStorage.setItem('bul_welcome_shown', 'true');
        }}
      />
      <ContextualHelp enabled={contextualHelpEnabled} />
      <UpdateChecker />
      <KeyboardNavigation />
            <DevTools />
            <NetworkMonitor />
            <SmartSuggestions />
            <NotificationSounds />
            <PerformanceOptimizer />
            <ToastContainer />
          </div>
        );
      }

