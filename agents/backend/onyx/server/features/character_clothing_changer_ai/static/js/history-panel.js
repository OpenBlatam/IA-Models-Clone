/**
 * History Panel Module
 * ====================
 * 
 * Manages the history panel display.
 */

const HistoryPanel = {
    /**
     * Render history panel
     * @param {HTMLElement} container - Container element
     */
    render(container) {
        if (!container) return;

        const history = History.getAll();

        if (history.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>📜 No hay historial aún.</p>
                    <p>Los procesamientos aparecerán aquí automáticamente.</p>
                </div>
            `;
            return;
        }

        let html = `
            <div class="history-controls">
                <button class="btn-secondary" onclick="HistoryPanel.clearAll()">🗑️ Limpiar Todo</button>
                <span class="history-count">${history.length} elementos</span>
            </div>
            <div class="history-list">
        `;

        history.forEach(item => {
            html += `
                <div class="history-item" data-id="${item.id}">
                    <div class="history-item-header">
                        <div class="history-item-info">
                            <h4>${item.characterName}</h4>
                            <p class="history-item-date">${this.formatDate(item.timestamp)}</p>
                        </div>
                        <div class="history-item-actions">
                            <button class="history-btn" onclick="HistoryPanel.viewItem('${item.id}')" title="Ver">👁️</button>
                            <button class="history-btn" onclick="HistoryPanel.deleteItem('${item.id}')" title="Eliminar">🗑️</button>
                        </div>
                    </div>
                    <div class="history-item-content">
                        <div class="history-item-images">
                            ${item.originalImage ? 
                                `<img src="${item.originalImage}" alt="Original" class="history-thumbnail">` :
                                '<div class="history-thumbnail-placeholder">Original</div>'
                            }
                            <span class="history-arrow">➡️</span>
                            ${item.resultImage ? 
                                `<img src="${item.resultImage}" alt="Result" class="history-thumbnail">` :
                                '<div class="history-thumbnail-placeholder">Resultado</div>'
                            }
                        </div>
                        <div class="history-item-details">
                            <p><strong>Descripción:</strong> ${item.clothingDescription}</p>
                            ${item.savedPath ? 
                                `<p><strong>Tensor:</strong> ${item.savedPath.split(/[/\\]/).pop()}</p>` : ''
                            }
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    },

    /**
     * View history item
     * @param {string} id - Item ID
     */
    viewItem(id) {
        const item = History.getById(id);
        if (item) {
            switchTab('comparison');
            Comparison.showItem(item);
        }
    },

    /**
     * Delete history item
     * @param {string} id - Item ID
     */
    deleteItem(id) {
        if (!confirm('¿Estás seguro de que deseas eliminar este elemento del historial?')) {
            return;
        }

        if (History.remove(id)) {
            this.render(document.getElementById('historyContent'));
            Gallery.render(document.getElementById('galleryContent'));
        }
    },

    /**
     * Clear all history
     */
    clearAll() {
        if (!confirm('¿Estás seguro de que deseas eliminar todo el historial? Esta acción no se puede deshacer.')) {
            return;
        }

        if (History.clear()) {
            this.render(document.getElementById('historyContent'));
            Gallery.render(document.getElementById('galleryContent'));
        }
    },

    /**
     * Format date for display
     * @param {string} timestamp - ISO timestamp
     * @returns {string} Formatted date
     */
    formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'Hace un momento';
        if (minutes < 60) return `Hace ${minutes} minuto${minutes > 1 ? 's' : ''}`;
        if (hours < 24) return `Hace ${hours} hora${hours > 1 ? 's' : ''}`;
        if (days < 7) return `Hace ${days} día${days > 1 ? 's' : ''}`;
        
        return date.toLocaleDateString('es-ES', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
};


