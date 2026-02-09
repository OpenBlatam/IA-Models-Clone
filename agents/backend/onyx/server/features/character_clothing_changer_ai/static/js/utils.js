/**
 * Utilities Module
 * ================
 * Utility functions for the application
 */

const Utils = {
    /**
     * Download image (delegates to FileDownloader)
     */
    downloadImage(imageSrc) {
        return FileDownloader.downloadImage(imageSrc);
    },

    /**
     * Export configuration (delegates to ConfigExporter)
     */
    exportConfig() {
        return ConfigExporter.export();
    }
};

// Global functions for onclick handlers
function downloadImage(imageSrc) {
    Utils.downloadImage(imageSrc);
}

function exportConfig() {
    Utils.exportConfig();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
}


