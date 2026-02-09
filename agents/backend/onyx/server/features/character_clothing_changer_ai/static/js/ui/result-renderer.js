/**
 * Result Renderer Module
 * ======================
 * Handles rendering of processing results
 */

const ResultRenderer = {
    /**
     * Render loading state
     */
    renderLoading(message = 'Procesando...') {
        return `
            <div class="loading">
                <div class="spinner"></div>
                <p>${message}</p>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
            </div>
        `;
    },

    /**
     * Render error message
     */
    renderError(message) {
        return `
            <div class="error-message">
                <h3>❌ Error</h3>
                <p><strong>${message}</strong></p>
            </div>
        `;
    },

    /**
     * Render success result
     */
    renderResult(data) {
        const resultImage = data.image_base64 || data.image_url || data.result_image;
        let html = '';
        
        if (resultImage) {
            html += `<img src="${resultImage}" class="result-image" alt="Resultado">`;
        }
        
        html += '<div class="result-info">';
        html += '<h3>✅ Procesamiento Completado</h3>';
        
        if (data.saved_path) {
            html += `<p><strong>Tensor guardado:</strong> ${data.saved_path}</p>`;
            const tensorId = data.saved_path.split('/').pop() || data.saved_path.split('\\').pop();
            html += `<a href="${CONFIG.API_BASE}/tensor/${tensorId}" class="download-btn" download>📥 Descargar Tensor</a>`;
        }
        
        if (resultImage) {
            html += `<button class="download-btn" onclick="downloadImage('${resultImage}')">💾 Descargar Imagen</button>`;
        }
        
        html += `<button class="download-btn" onclick="exportConfig()" style="background: #2196F3;">⚙️ Exportar Config</button>`;
        html += '</div>';
        
        return html;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResultRenderer;
}

