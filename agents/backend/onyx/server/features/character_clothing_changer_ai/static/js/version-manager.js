/**
 * Version Manager Module
 * ======================
 * Application versioning and migration
 */

const VersionManager = {
    /**
     * Current version
     */
    currentVersion: '1.0.0',
    
    /**
     * Version history
     */
    versionHistory: [],
    
    /**
     * Migrations
     */
    migrations: {},
    
    /**
     * Initialize version manager
     */
    init() {
        // Get stored version
        const storedVersion = typeof Storage !== 'undefined' 
            ? Storage.get('app_version') 
            : null;
        
        if (storedVersion && storedVersion !== this.currentVersion) {
            this.migrate(storedVersion, this.currentVersion);
        }
        
        // Save current version
        if (typeof Storage !== 'undefined') {
            Storage.save('app_version', this.currentVersion);
        }
        
        // Record version history
        this.versionHistory.push({
            version: this.currentVersion,
            timestamp: Date.now()
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Version manager initialized: ${this.currentVersion}`);
        }
    },
    
    /**
     * Register migration
     */
    registerMigration(fromVersion, toVersion, migrationFn) {
        const key = `${fromVersion}->${toVersion}`;
        this.migrations[key] = migrationFn;
    },
    
    /**
     * Migrate data between versions
     */
    async migrate(fromVersion, toVersion) {
        if (typeof Logger !== 'undefined') {
            Logger.info(`Migrating from ${fromVersion} to ${toVersion}`);
        }
        
        // Find migration path
        const migrationKey = `${fromVersion}->${toVersion}`;
        const migration = this.migrations[migrationKey];
        
        if (migration) {
            try {
                await migration();
                
                if (typeof Logger !== 'undefined') {
                    Logger.info(`Migration completed: ${migrationKey}`);
                }
                
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('version:migrated', { fromVersion, toVersion });
                }
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Migration failed: ${migrationKey}`, error);
                }
                if (typeof ErrorHandler !== 'undefined') {
                    ErrorHandler.handle(error, { context: 'version migration' });
                }
            }
        } else {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`No migration found for ${migrationKey}`);
            }
        }
    },
    
    /**
     * Get version info
     */
    getVersionInfo() {
        return {
            current: this.currentVersion,
            stored: typeof Storage !== 'undefined' 
                ? Storage.get('app_version') 
                : null,
            history: this.versionHistory
        };
    },
    
    /**
     * Check if update is available
     */
    async checkForUpdates() {
        // This would check with a server for new versions
        // For now, just return current version
        return {
            current: this.currentVersion,
            latest: this.currentVersion,
            updateAvailable: false
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionManager;
}

