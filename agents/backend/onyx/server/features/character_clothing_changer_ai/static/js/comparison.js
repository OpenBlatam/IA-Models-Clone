/**
 * Comparison Module
 * =================
 * Handles before/after comparison display with enhanced features
 */

const Comparison = {
    /**
     * Update comparison view
     */
    update(before, after, options = {}) {
        if (!before || !after) return;

        const comparisonContent = document.getElementById('comparisonContent');
        if (!comparisonContent) return;
        
        const {
            beforeLabel = 'Antes',
            afterLabel = 'Después',
            showDownload = true,
            showZoom = true
        } = options;
        
        comparisonContent.innerHTML = `
            <div class="comparison-item">
                <div class="comparison-label">${beforeLabel}</div>
                <img src="${before}" alt="${beforeLabel}" ${showZoom ? 'onclick="Comparison.zoomImage(this.src)"' : ''}>
                ${showDownload ? `<button class="download-btn-small" onclick="downloadImage('${before}')">💾</button>` : ''}
            </div>
            <div class="comparison-item">
                <div class="comparison-label">${afterLabel}</div>
                <img src="${after}" alt="${afterLabel}" ${showZoom ? 'onclick="Comparison.zoomImage(this.src)"' : ''}>
                ${showDownload ? `<button class="download-btn-small" onclick="downloadImage('${after}')">💾</button>` : ''}
            </div>
        `;
    },

    /**
     * Zoom image in modal
     */
    zoomImage(imageSrc) {
        if (typeof ModalViewer !== 'undefined') {
            ModalViewer.show(imageSrc);
        } else {
            // Fallback: open in new window
            window.open(imageSrc, '_blank');
        }
    },

    /**
     * Clear comparison view
     */
    clear() {
        const comparisonContent = document.getElementById('comparisonContent');
        if (comparisonContent) {
            comparisonContent.innerHTML = '<p class="empty-state">No hay comparación disponible aún.</p>';
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Comparison;
}
