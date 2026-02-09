/**
 * Backup Manager Component
 * @module robot-3d-view/controls/backup-manager
 */

'use client';

import { memo, useState, useEffect } from 'react';
import { backupManager, type BackupEntry } from '../utils/backup-restore';
import { notify } from '../utils/notifications';
import { formatBytes, formatDuration } from '../utils/ui-enhancements';

/**
 * Backup Manager Component
 * 
 * Provides UI for managing backups.
 * 
 * @returns Backup manager component
 */
export const BackupManager = memo(() => {
  const [isOpen, setIsOpen] = useState(false);
  const [backups, setBackups] = useState<BackupEntry[]>([]);
  const [stats, setStats] = useState(backupManager.getStats());

  useEffect(() => {
    if (!isOpen) return;

    const updateBackups = () => {
      setBackups(backupManager.getAllBackups());
      setStats(backupManager.getStats());
    };

    updateBackups();
    const interval = setInterval(updateBackups, 1000);
    return () => clearInterval(interval);
  }, [isOpen]);

  const handleCreateBackup = async (name: string, compress = false) => {
    try {
      // This would get current config from context
      const config = {} as any; // Replace with actual config
      const id = await backupManager.createBackup(name, config, compress);
      notify.success('Backup created successfully');
      setBackups(backupManager.getAllBackups());
    } catch (error) {
      notify.error('Failed to create backup');
    }
  };

  const handleRestore = (id: string) => {
    const backup = backupManager.restoreBackup(id);
    if (backup) {
      // This would restore config to context
      notify.success('Backup restored successfully');
    } else {
      notify.error('Failed to restore backup');
    }
  };

  const handleDelete = (id: string) => {
    backupManager.deleteBackup(id);
    notify.info('Backup deleted');
    setBackups(backupManager.getAllBackups());
  };

  const handleExport = async (id: string) => {
    try {
      const blob = await backupManager.exportBackup(id, true);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `backup-${id}.json.gz`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify.success('Backup exported');
    } catch (error) {
      notify.error('Failed to export backup');
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await backupManager.importBackup(file);
      notify.success('Backup imported successfully');
      setBackups(backupManager.getAllBackups());
    } catch (error) {
      notify.error('Failed to import backup');
    }

    // Reset input
    event.target.value = '';
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="absolute bottom-4 right-40 z-50 px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg"
        title="Backup Manager"
        aria-label="Open backup manager"
      >
        💾 Backup
      </button>
    );
  }

  return (
    <div
      className="absolute inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={() => setIsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Backup manager"
    >
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[80vh] overflow-y-auto shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Backup Manager</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close backup manager"
          >
            ✕
          </button>
        </div>

        {/* Statistics */}
        <div className="mb-4 p-4 bg-gray-700/50 rounded">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-400">Total Backups</div>
              <div className="text-white font-bold">{stats.total}</div>
            </div>
            <div>
              <div className="text-gray-400">Total Size</div>
              <div className="text-white font-bold">{formatBytes(stats.totalSize)}</div>
            </div>
            <div>
              <div className="text-gray-400">Average Size</div>
              <div className="text-white font-bold">{formatBytes(stats.averageSize)}</div>
            </div>
            <div>
              <div className="text-gray-400">Auto-backup</div>
              <div className="text-white font-bold">{stats.autoBackupEnabled ? 'Enabled' : 'Disabled'}</div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="mb-4 flex gap-2 flex-wrap">
          <button
            onClick={() => handleCreateBackup(`Manual backup ${new Date().toLocaleString()}`, false)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm transition-colors"
          >
            Create Backup
          </button>
          <button
            onClick={() => handleCreateBackup(`Compressed backup ${new Date().toLocaleString()}`, true)}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-white text-sm transition-colors"
          >
            Create Compressed Backup
          </button>
          <label className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-white text-sm transition-colors cursor-pointer">
            Import Backup
            <input
              type="file"
              accept=".json,.gz"
              onChange={handleImport}
              className="hidden"
            />
          </label>
          <button
            onClick={() => {
              backupManager.clear();
              notify.info('All backups cleared');
              setBackups([]);
            }}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-white text-sm transition-colors"
          >
            Clear All
          </button>
        </div>

        {/* Backups List */}
        <div className="space-y-2">
          {backups.length === 0 ? (
            <div className="text-gray-400 text-center py-8">No backups available</div>
          ) : (
            backups.map((backup) => (
              <div
                key={backup.id}
                className="p-4 bg-gray-700/50 rounded border border-gray-600"
              >
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <div className="font-semibold text-white">{backup.name}</div>
                    <div className="text-xs text-gray-400">
                      {new Date(backup.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">
                      {formatBytes(backup.size)}
                      {backup.compressed && ' (compressed)'}
                    </span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleRestore(backup.id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs transition-colors"
                  >
                    Restore
                  </button>
                  <button
                    onClick={() => handleExport(backup.id)}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white text-xs transition-colors"
                  >
                    Export
                  </button>
                  <button
                    onClick={() => handleDelete(backup.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-white text-xs transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
});

BackupManager.displayName = 'BackupManager';



