/**
 * Backup Manager Module
 * =====================
 * Data backup and restore functionality
 */

const BackupManager = {
    /**
     * Create backup of application data
     */
    async createBackup() {
        const backup = {
            version: '1.0',
            timestamp: new Date().toISOString(),
            data: {}
        };
        
        // Backup storage data
        if (typeof Storage !== 'undefined') {
            backup.data.storage = {
                history: Storage.getHistory(),
                gallery: Storage.getGallery(),
                theme: Storage.getTheme(),
                favorites: typeof Favorites !== 'undefined' ? Favorites.getAll() : []
            };
        }
        
        // Backup state
        if (typeof StateManager !== 'undefined') {
            backup.data.state = StateManager.getState();
        }
        
        // Backup config
        if (typeof ConfigManager !== 'undefined') {
            backup.data.config = ConfigManager.get('*');
        }
        
        // Backup analytics
        if (typeof Analytics !== 'undefined') {
            backup.data.analytics = Analytics.export();
        }
        
        return backup;
    },
    
    /**
     * Export backup as JSON
     */
    async exportBackup() {
        const backup = await this.createBackup();
        return JSON.stringify(backup, null, 2);
    },
    
    /**
     * Download backup file
     */
    async downloadBackup() {
        const backupJson = await this.exportBackup();
        const blob = new Blob([backupJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `clothing_changer_backup_${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
        
        if (typeof Notifications !== 'undefined') {
            Notifications.success('Backup descargado exitosamente');
        }
    },
    
    /**
     * Restore from backup
     */
    async restoreBackup(backupJson) {
        try {
            const backup = JSON.parse(backupJson);
            
            if (!backup.data) {
                throw new Error('Invalid backup format');
            }
            
            // Restore storage data
            if (backup.data.storage && typeof Storage !== 'undefined') {
                if (backup.data.storage.history) {
                    Storage.saveHistory(backup.data.storage.history);
                }
                if (backup.data.storage.gallery) {
                    Storage.saveGallery(backup.data.storage.gallery);
                }
                if (backup.data.storage.theme) {
                    Storage.saveTheme(backup.data.storage.theme);
                }
                if (backup.data.storage.favorites && typeof Favorites !== 'undefined') {
                    localStorage.setItem('favorites', JSON.stringify(backup.data.storage.favorites));
                    Favorites.init();
                }
            }
            
            // Restore state
            if (backup.data.state && typeof StateManager !== 'undefined') {
                StateManager.import(JSON.stringify(backup.data.state));
            }
            
            // Restore config
            if (backup.data.config && typeof ConfigManager !== 'undefined') {
                ConfigManager.import(JSON.stringify(backup.data.config));
            }
            
            // Reload UI
            if (typeof GalleryManager !== 'undefined') {
                GalleryManager.load();
            }
            if (typeof HistoryManager !== 'undefined') {
                HistoryManager.load();
            }
            
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Backup restaurado exitosamente');
            }
            
            // Emit restore event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('backup:restored', backup);
            }
            
            return true;
        } catch (error) {
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handle(error, { context: 'backup restore' });
            }
            return false;
        }
    },
    
    /**
     * Restore from file
     */
    async restoreFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
                const success = await this.restoreBackup(e.target.result);
                if (success) {
                    resolve(true);
                } else {
                    reject(new Error('Failed to restore backup'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    },
    
    /**
     * Get backup info
     */
    async getBackupInfo() {
        const backup = await this.createBackup();
        return {
            version: backup.version,
            timestamp: backup.timestamp,
            size: JSON.stringify(backup).length,
            items: {
                history: backup.data.storage?.history?.length || 0,
                gallery: backup.data.storage?.gallery?.length || 0,
                favorites: backup.data.storage?.favorites?.length || 0
            }
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BackupManager;
}

