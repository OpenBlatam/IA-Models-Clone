/**
 * Robot 3D View Client Component
 * 
 * Client-side component that handles 3D rendering and interactions.
 * Separated from container for better code splitting.
 * 
 * @module robot-3d-view/components/robot-3d-view-client
 */

'use client';

import dynamic from 'next/dynamic';
import { Canvas } from '@react-three/fiber';
import { PerspectiveCamera, Stats } from '../lib/drei-imports';
import { useSpring, animated } from '../lib/react-spring-imports';
import { useRobot3DView } from '../hooks/use-robot-3d-view';
import { useShortcuts } from '../hooks/use-shortcuts';
import { useRecording } from '../hooks/use-recording';
import { useHistory } from '../hooks/use-history';
import { usePlugins } from '../hooks/use-plugins';
import { pluginManager } from '../lib/plugin-system';
import { advancedLogger } from '../utils/logger-advanced';
import { metricsManager } from '../utils/metrics';
import { eventManager, Events } from '../utils/event-system';
import { telemetryManager } from '../utils/telemetry';
import { screenReaderManager } from '../utils/screen-reader';
import { backupManager } from '../utils/backup-restore';
import { trackEvent } from '../utils/analytics';
import { CANVAS_CONFIG } from '../constants';
import { notify } from '../utils/notifications';
import type { Robot3DViewProps } from '../schemas/validation-schemas';

// Dynamic imports for code splitting - only load 3D components on client
const Scene3D = dynamic(
  () => import('../scene/scene-3d').then((mod) => ({ default: mod.Scene3D })),
  {
    ssr: false,
    loading: () => <LoadingFallback />,
  }
);

const ViewControls = dynamic(
  () => import('../controls/view-controls').then((mod) => ({ default: mod.ViewControls })),
  { ssr: false }
);

const InfoOverlay = dynamic(
  () => import('../controls/info-overlay').then((mod) => ({ default: mod.InfoOverlay })),
  { ssr: false }
);

const InstructionsOverlay = dynamic(
  () => import('../controls/instructions-overlay').then((mod) => ({ default: mod.InstructionsOverlay })),
  { ssr: false }
);

const ScreenshotControls = dynamic(
  () => import('../controls/screenshot-controls').then((mod) => ({ default: mod.ScreenshotControls })),
  { ssr: false }
);

const PresetSelector = dynamic(
  () => import('../controls/preset-selector').then((mod) => ({ default: mod.PresetSelector })),
  { ssr: false }
);

const ThemeSelector = dynamic(
  () => import('../controls/theme-selector').then((mod) => ({ default: mod.ThemeSelector })),
  { ssr: false }
);

const NotificationToast = dynamic(
  () => import('./notification-toast').then((mod) => ({ default: mod.NotificationToast })),
  { ssr: false }
);

const HelpOverlay = dynamic(
  () => import('./help-overlay').then((mod) => ({ default: mod.HelpOverlay })),
  { ssr: false }
);

const ConfigManager = dynamic(
  () => import('../controls/config-manager').then((mod) => ({ default: mod.ConfigManager })),
  { ssr: false }
);

const HistoryControls = dynamic(
  () => import('../controls/history-controls').then((mod) => ({ default: mod.HistoryControls })),
  { ssr: false }
);

const RecordingControls = dynamic(
  () => import('../controls/recording-controls').then((mod) => ({ default: mod.RecordingControls })),
  { ssr: false }
);

const WidgetManager = dynamic(
  () => import('../controls/widget-manager').then((mod) => ({ default: mod.WidgetManager })),
  { ssr: false }
);

const CommandPalette = dynamic(
  () => import('../controls/command-palette').then((mod) => ({ default: mod.CommandPalette })),
  { ssr: false }
);

const TutorialOverlay = dynamic(
  () => import('./tutorial-overlay').then((mod) => ({ default: mod.TutorialOverlay })),
  { ssr: false }
);

const LogViewer = dynamic(
  () => import('./log-viewer').then((mod) => ({ default: mod.LogViewer })),
  { ssr: false }
);

const LanguageSelector = dynamic(
  () => import('../controls/language-selector').then((mod) => ({ default: mod.LanguageSelector })),
  { ssr: false }
);

const MetricsPanel = dynamic(
  () => import('../controls/metrics-panel').then((mod) => ({ default: mod.MetricsPanel })),
  { ssr: false }
);

const BackupManager = dynamic(
  () => import('../controls/backup-manager').then((mod) => ({ default: mod.BackupManager })),
  { ssr: false }
);

// Lazy load LoadingFallback
const LoadingFallback = dynamic(
  () => import('./loading-fallback').then((mod) => ({ default: mod.LoadingFallback })),
  { ssr: false }
);

/**
 * Robot 3D View Client Component
 * 
 * Handles all client-side 3D rendering logic.
 * Refactored to use consolidated hook for cleaner code.
 * 
 * @param props - Component props
 * @returns Robot 3D view client component
 */
export function Robot3DViewClient({ fullscreen }: Pick<Robot3DViewProps, 'fullscreen'>) {
  // Consolidated hook for all state and actions
  const {
    currentPos,
    targetPos,
    trajectory,
    config,
    viewportQuality,
    toggleStats,
    toggleGizmo,
    toggleStars,
    toggleWaypoints,
    toggleGrid,
    toggleObjects,
    toggleAutoRotate,
    setCameraPreset,
    status,
  } = useRobot3DView(fullscreen);

  // History management
  const { addEntry } = useHistory();

  // Recording management
  const { isRecording, recordFrame } = useRecording();

  // Plugin management
  const { getEnabledPlugins } = usePlugins();

  // Initialize plugins
  useEffect(() => {
    pluginManager.initialize();
    advancedLogger.info('Robot 3D View initialized', { config }, 'initialization');
    metricsManager.increment('app.initializations');
    eventManager.emit(Events.INITIALIZED, { config });
    telemetryManager.track('view:initialized', { config });
    screenReaderManager.announce('Robot 3D View loaded');
    return () => {
      pluginManager.cleanup();
      advancedLogger.info('Robot 3D View cleaned up', {}, 'cleanup');
      eventManager.emit(Events.DESTROYED, {});
      telemetryManager.track('view:destroyed', {});
    };
  }, []);

  // Notify plugins of config changes
  useEffect(() => {
    pluginManager.notifyConfigChange(config);
    advancedLogger.debug('Configuration changed', { config }, 'config');
    metricsManager.increment('config.changes');
    eventManager.emit(Events.CONFIG_CHANGED, config);
    
    // Auto-backup on config change
    backupManager.createBackup(
      `Auto-backup ${new Date().toISOString()}`,
      config,
      true
    ).catch((error) => {
      advancedLogger.warn('Auto-backup failed', { error }, 'backup');
    });
  }, [config]);

  // Track frame when recording
  useEffect(() => {
    if (isRecording) {
      recordFrame(currentPos, targetPos, config);
    }
  }, [isRecording, currentPos, targetPos, config, recordFrame]);

  // Keyboard shortcuts
  useShortcuts({
    enabled: true,
    onShortcut: (action) => {
      trackEvent('shortcut-used', { action });

      switch (action) {
        case 'toggle-stats':
          toggleStats();
          addEntry(config, 'Toggle stats');
          break;
        case 'toggle-gizmo':
          toggleGizmo();
          addEntry(config, 'Toggle gizmo');
          break;
        case 'toggle-grid':
          toggleGrid();
          addEntry(config, 'Toggle grid');
          break;
        case 'toggle-objects':
          toggleObjects();
          addEntry(config, 'Toggle objects');
          break;
        case 'toggle-auto-rotate':
          toggleAutoRotate();
          addEntry(config, 'Toggle auto-rotate');
          break;
        case 'toggle-stars':
          toggleStars();
          addEntry(config, 'Toggle stars');
          break;
        case 'toggle-waypoints':
          toggleWaypoints();
          addEntry(config, 'Toggle waypoints');
          break;
        case 'camera-front':
          setCameraPreset('front');
          addEntry(config, 'Camera: front');
          break;
        case 'camera-top':
          setCameraPreset('top');
          addEntry(config, 'Camera: top');
          break;
        case 'camera-side':
          setCameraPreset('side');
          addEntry(config, 'Camera: side');
          break;
        case 'camera-iso':
          setCameraPreset('iso');
          addEntry(config, 'Camera: iso');
          break;
        case 'reset-camera':
          setCameraPreset(null);
          addEntry(config, 'Camera: reset');
          break;
        case 'undo':
          // Handled by HistoryControls
          break;
        case 'redo':
          // Handled by HistoryControls
          break;
        case 'start-recording':
          // Handled by RecordingControls
          break;
        case 'stop-recording':
          // Handled by RecordingControls
          break;
        default:
          break;
      }
    },
  });

  // Animation for info panel
  const [infoPanelStyle] = useSpring(() => ({
    from: { opacity: 0, transform: 'translateY(-10px)' },
    to: { opacity: 1, transform: 'translateY(0px)' },
    config: { tension: 280, friction: 60 },
  }));

  return (
    <>
      <Canvas
        camera={CANVAS_CONFIG.camera}
        gl={{
          ...CANVAS_CONFIG.gl,
          antialias: viewportQuality.antialias,
        }}
        dpr={viewportQuality.dpr}
        shadows={viewportQuality.shadows}
      >
        <PerspectiveCamera
          makeDefault
          position={CANVAS_CONFIG.camera.position}
          fov={CANVAS_CONFIG.camera.fov}
        />

        <Scene3D
          trajectory={trajectory}
          currentPos={currentPos}
          targetPos={targetPos}
          config={config}
        />

        {config.showStats && <Stats />}
      </Canvas>

      <animated.div style={infoPanelStyle}>
        <InfoOverlay
          currentPos={currentPos}
          targetPos={targetPos}
          status={status}
        />
      </animated.div>

      <ViewControls
        config={config}
        onToggleStats={toggleStats}
        onToggleGizmo={toggleGizmo}
        onToggleStars={toggleStars}
        onToggleWaypoints={toggleWaypoints}
        onToggleGrid={toggleGrid}
        onToggleObjects={toggleObjects}
        onToggleAutoRotate={toggleAutoRotate}
        onSetCameraPreset={setCameraPreset}
      />

      <InstructionsOverlay />

      <ScreenshotControls />

      <PresetSelector
        onPresetChange={(config) => {
          // Apply preset configuration
          Object.entries(config).forEach(([key, value]) => {
            if (key === 'showStats') toggleStats();
            else if (key === 'showGizmo') toggleGizmo();
            else if (key === 'showStars') toggleStars();
            else if (key === 'showWaypoints') toggleWaypoints();
            else if (key === 'showGrid') toggleGrid();
            else if (key === 'showObjects') toggleObjects();
            else if (key === 'autoRotate') toggleAutoRotate();
            else if (key === 'cameraPreset') setCameraPreset(value);
          });
        }}
      />

      <ThemeSelector />

      <ConfigManager
        config={config}
        onConfigChange={(newConfig) => {
          // Apply new configuration
          Object.entries(newConfig).forEach(([key, value]) => {
            if (key === 'showStats' && value !== config.showStats) toggleStats();
            else if (key === 'showGizmo' && value !== config.showGizmo) toggleGizmo();
            else if (key === 'showStars' && value !== config.showStars) toggleStars();
            else if (key === 'showWaypoints' && value !== config.showWaypoints) toggleWaypoints();
            else if (key === 'showGrid' && value !== config.showGrid) toggleGrid();
            else if (key === 'showObjects' && value !== config.showObjects) toggleObjects();
            else if (key === 'autoRotate' && value !== config.autoRotate) toggleAutoRotate();
            else if (key === 'cameraPreset' && value !== config.cameraPreset) setCameraPreset(value);
          });
          notify.success('Configuración aplicada');
        }}
      />

      <HistoryControls
        onConfigChange={(newConfig) => {
          // Apply configuration from history
          Object.entries(newConfig).forEach(([key, value]) => {
            if (key === 'showStats' && value !== config.showStats) toggleStats();
            else if (key === 'showGizmo' && value !== config.showGizmo) toggleGizmo();
            else if (key === 'showStars' && value !== config.showStars) toggleStars();
            else if (key === 'showWaypoints' && value !== config.showWaypoints) toggleWaypoints();
            else if (key === 'showGrid' && value !== config.showGrid) toggleGrid();
            else if (key === 'showObjects' && value !== config.showObjects) toggleObjects();
            else if (key === 'autoRotate' && value !== config.autoRotate) toggleAutoRotate();
            else if (key === 'cameraPreset' && value !== config.cameraPreset) setCameraPreset(value);
          });
        }}
      />

      <RecordingControls />

      <WidgetManager />

      <CommandPalette />

      <TutorialOverlay />

      <LogViewer />

      <LanguageSelector />

      <MetricsPanel />

      <BackupManager />

      <HelpOverlay />

      <NotificationToast />
    </>
  );
}

