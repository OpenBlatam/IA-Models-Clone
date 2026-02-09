/**
 * Statistics Module
 * =================
 * Handles statistics and metrics with enhanced features
 */

const Stats = {
    /**
     * Get processing statistics
     */
    getStats() {
        const history = typeof HistoryManager !== 'undefined' ? HistoryManager.items : [];
        const gallery = typeof GalleryManager !== 'undefined' ? GalleryManager.items : [];
        const favorites = typeof Favorites !== 'undefined' ? Favorites.getAll() : [];

        return {
            totalProcessed: history.length,
            totalInGallery: gallery.length,
            totalFavorites: favorites.length,
            lastProcessed: history.length > 0 ? history[0].timestamp : null,
            mostUsedDescription: this.getMostUsedDescription(history),
            averageProcessingTime: this.calculateAverageTime(history),
            uniqueCharacters: this.getUniqueCharacters(history),
            processingRate: this.calculateProcessingRate(history)
        };
    },

    /**
     * Get most used clothing description
     */
    getMostUsedDescription(history) {
        const frequency = StatsCalculator.calculateFrequency(
            history,
            'clothingDescription'
        );
        const mostFrequent = StatsCalculator.getMostFrequent(frequency);
        return {
            description: mostFrequent.value || 'N/A',
            count: mostFrequent.count
        };
    },

    /**
     * Calculate average processing time (placeholder)
     */
    calculateAverageTime(history) {
        // This would require tracking actual processing times
        if (!history || history.length === 0) return 'N/A';
        
        // Try to extract from quality_metrics if available
        const times = history
            .filter(item => item.quality_metrics && item.quality_metrics.processing_time)
            .map(item => item.quality_metrics.processing_time);
        
        if (times.length === 0) return 'N/A';
        
        const avg = times.reduce((sum, time) => sum + time, 0) / times.length;
        return `${(avg / 1000).toFixed(2)}s`;
    },

    /**
     * Get unique characters count
     */
    getUniqueCharacters(history) {
        if (!history || history.length === 0) return 0;
        
        const characters = new Set(
            history
                .filter(item => item.character_name)
                .map(item => item.character_name)
        );
        
        return characters.size;
    },

    /**
     * Calculate processing rate (items per day)
     */
    calculateProcessingRate(history) {
        if (!history || history.length === 0) return 0;
        
        const now = new Date();
        const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        
        const recentItems = history.filter(item => {
            const itemDate = new Date(item.timestamp);
            return itemDate >= oneDayAgo;
        });
        
        return recentItems.length;
    },

    /**
     * Display statistics
     */
    display() {
        const stats = this.getStats();
        const lastProcessedDate = stats.lastProcessed 
            ? new Date(stats.lastProcessed).toLocaleString() 
            : 'N/A';
        
        const statsHTML = `
            <div class="stats-panel">
                <h3>📊 Estadísticas</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${stats.totalProcessed}</div>
                        <div class="stat-label">Procesamientos</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.totalInGallery}</div>
                        <div class="stat-label">En Galería</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.totalFavorites}</div>
                        <div class="stat-label">Favoritos</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.uniqueCharacters}</div>
                        <div class="stat-label">Personajes Únicos</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.processingRate}</div>
                        <div class="stat-label">Últimas 24h</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.mostUsedDescription.count}</div>
                        <div class="stat-label">Más usado: "${(stats.mostUsedDescription.description || '').substring(0, 20)}..."</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.averageProcessingTime}</div>
                        <div class="stat-label">Tiempo Promedio</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${lastProcessedDate}</div>
                        <div class="stat-label">Último Procesamiento</div>
                    </div>
                </div>
            </div>
        `;
        return statsHTML;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Stats;
}


