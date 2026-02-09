/**
 * Backup Manager Module
 * =====================
 * Manages data backup and restore functionality
 */

const BackupManager = {
    /**
     * Backup data structure
     */
    backupData: {
        version: '1.0.0',
        timestamp: null,
        data: {}
    },
    
    /**
     * Initialize backup manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Backup manager initialized');
        }
    },
    
    /**
     * Create backup
     */
    createBackup() {
        try {
            const backup = {
                version: typeof VersionManager !== 'undefined' 
                    ? VersionManager.getCurrentVersion() 
                    : '1.0.0',
                timestamp: new Date().toISOString(),
                data: {
                    history: this.backupHistory(),
                    gallery: this.backupGallery(),
                    favorites: this.backupFavorites(),
                    settings: this.backupSettings(),
                    config: this.backupConfig()
                }
            };
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Backup created', { timestamp: backup.timestamp });
            }
            
            // Emit backup created event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('backup:created', { timestamp: backup.timestamp });
            }
            
            return backup;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Failed to create backup', error);
            }
            throw error;
        }
    },
    
    /**
     * Backup history
     */
    backupHistory() {
        try {
            if (typeof HistoryManager !== 'undefined') {
                return HistoryManager.getAll();
            }
            
            // Fallback to localStorage
            const stored = localStorage.getItem('processing_history');
            return stored ? JSON.parse(stored) : [];
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to backup history', error);
            }
            return [];
        }
    },
    
    /**
     * Backup gallery
     */
    backupGallery() {
        try {
            if (typeof GalleryManager !== 'undefined') {
                return GalleryManager.getAll();
            }
            
            // Fallback to localStorage
            const stored = localStorage.getItem('gallery_items');
            return stored ? JSON.parse(stored) : [];
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to backup gallery', error);
            }
            return [];
        }
    },
    
    /**
     * Backup favorites
     */
    backupFavorites() {
        try {
            if (typeof Favorites !== 'undefined') {
                return Favorites.getAll();
            }
            
            // Fallback to localStorage
            const stored = localStorage.getItem('favorites');
            return stored ? JSON.parse(stored) : [];
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to backup favorites', error);
            }
            return [];
        }
    },
    
    /**
     * Backup settings
     */
    backupSettings() {
        try {
            const settings = {};
            
            // Theme
            const theme = localStorage.getItem('theme');
            if (theme) settings.theme = theme;
            
            // Language
            const language = localStorage.getItem('app_language');
            if (language) settings.language = language;
            
            // Other settings
            if (typeof ConfigManager !== 'undefined') {
                settings.config = ConfigManager.get();
            }
            
            return settings;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to backup settings', error);
            }
            return {};
        }
    },
    
    /**
     * Backup config
     */
    backupConfig() {
        try {
            if (typeof ConfigManager !== 'undefined') {
                return ConfigManager.get();
            }
            return {};
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to backup config', error);
            }
            return {};
        }
    },
    
    /**
     * Restore backup
     */
    async restoreBackup(backup, options = {}) {
        const {
            clearExisting = false,
            merge = true
        } = options;
        
        try {
            if (!backup || !backup.data) {
                throw new Error('Invalid backup data');
            }
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Restoring backup', { timestamp: backup.timestamp });
            }
            
            // Emit restore start event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('backup:restore_start', { timestamp: backup.timestamp });
            }
            
            // Clear existing data if requested
            if (clearExisting) {
                this.clearAllData();
            }
            
            // Restore history
            if (backup.data.history) {
                this.restoreHistory(backup.data.history, merge);
            }
            
            // Restore gallery
            if (backup.data.gallery) {
                this.restoreGallery(backup.data.gallery, merge);
            }
            
            // Restore favorites
            if (backup.data.favorites) {
                this.restoreFavorites(backup.data.favorites, merge);
            }
            
            // Restore settings
            if (backup.data.settings) {
                this.restoreSettings(backup.data.settings);
            }
            
            // Restore config
            if (backup.data.config) {
                this.restoreConfig(backup.data.config);
            }
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Backup restored successfully');
            }
            
            // Emit restore complete event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('backup:restore_complete', { timestamp: backup.timestamp });
            }
            
            // Show notification
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Backup restaurado exitosamente');
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Failed to restore backup', error);
            }
            
            if (typeof Notifications !== 'undefined') {
                Notifications.error('Error al restaurar el backup');
            }
            
            throw error;
        }
    },
    
    /**
     * Restore history
     */
    restoreHistory(history, merge) {
        if (typeof HistoryManager !== 'undefined') {
            if (merge) {
                history.forEach(item => HistoryManager.add(item));
            } else {
                HistoryManager.clear();
                history.forEach(item => HistoryManager.add(item));
            }
        } else {
            // Fallback to localStorage
            if (merge) {
                const existing = JSON.parse(localStorage.getItem('processing_history') || '[]');
                const merged = [...existing, ...history];
                localStorage.setItem('processing_history', JSON.stringify(merged));
            } else {
                localStorage.setItem('processing_history', JSON.stringify(history));
            }
        }
    },
    
    /**
     * Restore gallery
     */
    restoreGallery(gallery, merge) {
        if (typeof GalleryManager !== 'undefined') {
            if (merge) {
                gallery.forEach(item => GalleryManager.add(item));
            } else {
                GalleryManager.clear();
                gallery.forEach(item => GalleryManager.add(item));
            }
        } else {
            // Fallback to localStorage
            if (merge) {
                const existing = JSON.parse(localStorage.getItem('gallery_items') || '[]');
                const merged = [...existing, ...gallery];
                localStorage.setItem('gallery_items', JSON.stringify(merged));
            } else {
                localStorage.setItem('gallery_items', JSON.stringify(gallery));
            }
        }
    },
    
    /**
     * Restore favorites
     */
    restoreFavorites(favorites, merge) {
        if (typeof Favorites !== 'undefined') {
            if (merge) {
                favorites.forEach(item => Favorites.add(item));
            } else {
                Favorites.clear();
                favorites.forEach(item => Favorites.add(item));
            }
        } else {
            // Fallback to localStorage
            if (merge) {
                const existing = JSON.parse(localStorage.getItem('favorites') || '[]');
                const merged = [...existing, ...favorites];
                localStorage.setItem('favorites', JSON.stringify(merged));
            } else {
                localStorage.setItem('favorites', JSON.stringify(favorites));
            }
        }
    },
    
    /**
     * Restore settings
     */
    restoreSettings(settings) {
        if (settings.theme) {
            localStorage.setItem('theme', settings.theme);
            if (typeof UI !== 'undefined') {
                UI.setTheme(settings.theme);
            }
        }
        
        if (settings.language) {
            localStorage.setItem('app_language', settings.language);
            if (typeof I18n !== 'undefined') {
                I18n.setLanguage(settings.language);
            }
        }
        
        if (settings.config && typeof ConfigManager !== 'undefined') {
            ConfigManager.update(settings.config);
        }
    },
    
    /**
     * Restore config
     */
    restoreConfig(config) {
        if (typeof ConfigManager !== 'undefined') {
            ConfigManager.update(config);
        }
    },
    
    /**
     * Clear all data
     */
    clearAllData() {
        // Clear history
        if (typeof HistoryManager !== 'undefined') {
            HistoryManager.clear();
        } else {
            localStorage.removeItem('processing_history');
        }
        
        // Clear gallery
        if (typeof GalleryManager !== 'undefined') {
            GalleryManager.clear();
        } else {
            localStorage.removeItem('gallery_items');
        }
        
        // Clear favorites
        if (typeof Favorites !== 'undefined') {
            Favorites.clear();
        } else {
            localStorage.removeItem('favorites');
        }
    },
    
    /**
     * Export backup to file
     */
    exportBackup(backup = null) {
        const data = backup || this.createBackup();
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `backup_${data.timestamp.replace(/[:.]/g, '-')}.json`;
        link.click();
        URL.revokeObjectURL(url);
        
        if (typeof Notifications !== 'undefined') {
            Notifications.success('Backup descargado exitosamente');
        }
    },
    
    /**
     * Import backup from file
     */
    async importBackup(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = async (e) => {
                try {
                    const backup = JSON.parse(e.target.result);
                    await this.restoreBackup(backup);
                    resolve(backup);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsText(file);
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BackupManager;
}

