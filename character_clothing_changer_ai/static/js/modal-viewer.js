/**
 * Modal Viewer Module
 * ===================
 * Modal dialog system for viewing images and details
 */

const ModalViewer = {
    /**
     * Current modal element
     */
    currentModal: null,
    
    /**
     * Show modal with item data
     */
    show(item) {
        // Close existing modal if any
        this.close();
        
        // Create modal element
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.id = 'itemModal';
        
        const imageUrl = item.image_base64 || item.image_url || item.result_image || '';
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>${item.character_name || 'Resultado'}</h2>
                    <button class="modal-close" onclick="ModalViewer.close()" aria-label="Cerrar">×</button>
                </div>
                <div class="modal-body">
                    ${imageUrl ? `
                        <div class="modal-image">
                            <img src="${imageUrl}" alt="${item.character_name || 'Resultado'}" loading="lazy">
                        </div>
                    ` : '<div class="no-image">Sin imagen disponible</div>'}
                    <div class="modal-info">
                        <div class="info-section">
                            <h3>Descripción de Ropa</h3>
                            <p>${item.clothing_description || 'Sin descripción'}</p>
                        </div>
                        ${item.prompt_used ? `
                            <div class="info-section">
                                <h3>Prompt Usado</h3>
                                <p>${item.prompt_used}</p>
                            </div>
                        ` : ''}
                        ${item.negative_prompt_used ? `
                            <div class="info-section">
                                <h3>Prompt Negativo</h3>
                                <p>${item.negative_prompt_used}</p>
                            </div>
                        ` : ''}
                        ${item.quality_metrics ? `
                            <div class="info-section">
                                <h3>Métricas de Calidad</h3>
                                <div class="metrics">
                                    ${Object.entries(item.quality_metrics).map(([key, value]) => `
                                        <div class="metric-item">
                                            <span class="metric-label">${key}:</span>
                                            <span class="metric-value">${typeof value === 'number' ? value.toFixed(3) : value}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                        ${item.timestamp ? `
                            <div class="info-section">
                                <h3>Fecha</h3>
                                <p>${new Date(item.timestamp).toLocaleString()}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn" onclick="ModalViewer.download()">💾 Descargar</button>
                    ${item.saved_path ? `
                        <a href="${CONFIG.API_BASE}/tensor/${item.saved_path.split('/').pop()}" 
                           class="btn" 
                           download>
                            📥 Descargar Tensor
                        </a>
                    ` : ''}
                    <button class="btn btn-secondary" onclick="ModalViewer.close()">Cerrar</button>
                </div>
            </div>
        `;
        
        // Store current item for download
        modal.dataset.item = JSON.stringify(item);
        
        // Add to DOM
        document.body.appendChild(modal);
        this.currentModal = modal;
        
        // Add event listeners
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.close();
            }
        });
        
        // Close on Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                this.close();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
        
        // Prevent body scroll
        document.body.style.overflow = 'hidden';
        
        // Log modal open
        if (typeof Logger !== 'undefined') {
            Logger.debug('Modal opened', { characterName: item.character_name });
        }
    },
    
    /**
     * Close modal
     */
    close() {
        if (this.currentModal) {
            this.currentModal.remove();
            this.currentModal = null;
            document.body.style.overflow = '';
            
            if (typeof Logger !== 'undefined') {
                Logger.debug('Modal closed');
            }
        }
    },
    
    /**
     * Download current item
     */
    download() {
        if (!this.currentModal) {
            return;
        }
        
        try {
            const item = JSON.parse(this.currentModal.dataset.item);
            const imageUrl = item.image_base64 || item.image_url || item.result_image;
            
            if (!imageUrl) {
                if (typeof Notifications !== 'undefined') {
                    Notifications.error('No hay imagen para descargar');
                }
                return;
            }
            
            // Create download link
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = `clothing-change-${item.character_name || 'result'}-${Date.now()}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Imagen descargada');
            }
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Item downloaded from modal', { characterName: item.character_name });
            }
        } catch (error) {
            console.error('Error downloading from modal:', error);
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handleApiError(error, 'Modal Download');
            }
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModalViewer;
}
