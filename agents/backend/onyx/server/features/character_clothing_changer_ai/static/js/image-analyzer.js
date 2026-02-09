/**
 * Image Analyzer Module
 * =====================
 * Analyzes uploaded images and displays statistics
 */

const ImageAnalyzer = {
    /**
     * Analyze image and display statistics
     */
    analyze(imageSrc) {
        const img = new Image();
        img.onload = () => {
            const stats = ImageStatsCalculator.calculate(img);
            const recommendations = ImageStatsCalculator.getRecommendations(stats);
            this.displayStats(stats, recommendations);
        };
        img.onerror = () => {
            console.error('Error loading image for analysis');
        };
        img.src = imageSrc;
    },

    /**
     * Display image statistics
     */
    displayStats(stats, recommendations = []) {
        const analysisContainer = document.getElementById('imageAnalysis');
        if (!analysisContainer) return;
        
        let html = `
            <div class="image-stats">
                <div class="stat-card">
                    <div class="stat-value">${stats.width}</div>
                    <div class="stat-label">Ancho (px)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.height}</div>
                    <div class="stat-label">Alto (px)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.aspectRatio}</div>
                    <div class="stat-label">Aspecto</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.megapixels}</div>
                    <div class="stat-label">MP</div>
                </div>
            </div>
        `;
        
        if (recommendations.length > 0) {
            html += `
                <div class="recommendations" style="margin-top: 10px; padding: 10px; background: #fff3cd; border-radius: 5px; font-size: 12px;">
                    ${recommendations.map(rec => `<p style="margin: 5px 0;">⚠️ ${rec}</p>`).join('')}
                </div>
            `;
        }
        
        analysisContainer.innerHTML = html;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageAnalyzer;
}


