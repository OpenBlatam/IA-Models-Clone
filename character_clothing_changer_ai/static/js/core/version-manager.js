/**
 * Version Manager Module
 * =====================
 * Manages application versioning, updates, and migration
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
     * Migration handlers
     */
    migrations: new Map(),
    
    /**
     * Initialize version manager
     */
    init() {
        // Load version from storage
        const storedVersion = localStorage.getItem('app_version');
        
        if (storedVersion && storedVersion !== this.currentVersion) {
            // Version changed, run migrations
            this.runMigrations(storedVersion, this.currentVersion);
        }
        
        // Save current version
        localStorage.setItem('app_version', this.currentVersion);
        
        // Add to history
        this.versionHistory.push({
            version: this.currentVersion,
            timestamp: new Date().toISOString()
        });
        
        // Keep only last 10 versions
        if (this.versionHistory.length > 10) {
            this.versionHistory.shift();
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Version manager initialized', { version: this.currentVersion });
        }
        
        // Emit version event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('version:initialized', { version: this.currentVersion });
        }
    },
    
    /**
     * Register a migration
     */
    registerMigration(fromVersion, toVersion, handler) {
        const key = `${fromVersion}->${toVersion}`;
        this.migrations.set(key, handler);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Migration registered: ${key}`);
        }
    },
    
    /**
     * Run migrations
     */
    async runMigrations(fromVersion, toVersion) {
        if (typeof Logger !== 'undefined') {
            Logger.info(`Running migrations from ${fromVersion} to ${toVersion}`);
        }
        
        const migrationsToRun = this.getMigrationsBetween(fromVersion, toVersion);
        
        for (const migration of migrationsToRun) {
            try {
                const key = `${migration.from}->${migration.to}`;
                const handler = this.migrations.get(key);
                
                if (handler) {
                    if (typeof Logger !== 'undefined') {
                        Logger.info(`Running migration: ${key}`);
                    }
                    
                    await handler();
                    
                    if (typeof Logger !== 'undefined') {
                        Logger.info(`Migration completed: ${key}`);
                    }
                }
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Migration failed: ${migration.from}->${migration.to}`, error);
                }
                
                // Emit migration error event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('version:migration_error', { 
                        from: migration.from, 
                        to: migration.to, 
                        error 
                    });
                }
            }
        }
        
        // Emit migration complete event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('version:migrations_complete', { fromVersion, toVersion });
        }
    },
    
    /**
     * Get migrations between versions
     */
    getMigrationsBetween(fromVersion, toVersion) {
        const migrations = [];
        const from = this.parseVersion(fromVersion);
        const to = this.parseVersion(toVersion);
        
        // Simple version comparison (can be enhanced)
        if (this.compareVersions(fromVersion, toVersion) < 0) {
            // Upgrade path
            this.migrations.forEach((handler, key) => {
                const [from, to] = key.split('->');
                if (this.compareVersions(fromVersion, from) <= 0 && 
                    this.compareVersions(to, toVersion) <= 0) {
                    migrations.push({ from, to, handler });
                }
            });
        }
        
        return migrations;
    },
    
    /**
     * Parse version string
     */
    parseVersion(version) {
        const parts = version.split('.').map(Number);
        return {
            major: parts[0] || 0,
            minor: parts[1] || 0,
            patch: parts[2] || 0
        };
    },
    
    /**
     * Compare two versions
     */
    compareVersions(v1, v2) {
        const parsed1 = this.parseVersion(v1);
        const parsed2 = this.parseVersion(v2);
        
        if (parsed1.major !== parsed2.major) {
            return parsed1.major - parsed2.major;
        }
        if (parsed1.minor !== parsed2.minor) {
            return parsed1.minor - parsed2.minor;
        }
        return parsed1.patch - parsed2.patch;
    },
    
    /**
     * Get current version
     */
    getCurrentVersion() {
        return this.currentVersion;
    },
    
    /**
     * Get version history
     */
    getVersionHistory() {
        return [...this.versionHistory];
    },
    
    /**
     * Check if update is available
     */
    async checkForUpdates() {
        try {
            // Check for updates from server
            if (typeof API !== 'undefined') {
                const response = await API.get('/version');
                if (response.success && response.data.version) {
                    const latestVersion = response.data.version;
                    const comparison = this.compareVersions(this.currentVersion, latestVersion);
                    
                    if (comparison < 0) {
                        // Update available
                        if (typeof EventBus !== 'undefined') {
                            EventBus.emit('version:update_available', { 
                                current: this.currentVersion, 
                                latest: latestVersion 
                            });
                        }
                        
                        return {
                            available: true,
                            current: this.currentVersion,
                            latest: latestVersion
                        };
                    }
                }
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to check for updates', error);
            }
        }
        
        return {
            available: false,
            current: this.currentVersion
        };
    },
    
    /**
     * Get version info
     */
    getVersionInfo() {
        return {
            current: this.currentVersion,
            history: this.versionHistory,
            migrations: Array.from(this.migrations.keys())
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionManager;
}

