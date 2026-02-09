/**
 * Image Stats Calculator Module
 * =============================
 * Calculates image statistics and metrics
 */

const ImageStatsCalculator = {
    /**
     * Calculate image statistics
     */
    calculate(image) {
        return {
            width: image.width,
            height: image.height,
            aspectRatio: this.calculateAspectRatio(image.width, image.height),
            megapixels: this.calculateMegapixels(image.width, image.height),
            totalPixels: image.width * image.height,
            isLandscape: image.width > image.height,
            isPortrait: image.height > image.width,
            isSquare: image.width === image.height
        };
    },

    /**
     * Calculate aspect ratio
     */
    calculateAspectRatio(width, height) {
        return (width / height).toFixed(2);
    },

    /**
     * Calculate megapixels
     */
    calculateMegapixels(width, height) {
        return ((width * height) / 1000000).toFixed(2);
    },

    /**
     * Get image format recommendations
     */
    getRecommendations(stats) {
        const recommendations = [];
        
        if (stats.width < 512 || stats.height < 512) {
            recommendations.push('La imagen es pequeña. Se recomienda usar imágenes de al menos 512x512px para mejores resultados.');
        }
        
        if (stats.megapixels > 10) {
            recommendations.push('La imagen es muy grande. El procesamiento puede tardar más tiempo.');
        }
        
        if (stats.aspectRatio < 0.5 || stats.aspectRatio > 2) {
            recommendations.push('La relación de aspecto es extrema. Puede afectar la calidad del resultado.');
        }
        
        return recommendations;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageStatsCalculator;
}

