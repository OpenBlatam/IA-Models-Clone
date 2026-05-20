/**
 * Item Renderer Module
 * ====================
 * Centralized rendering utilities for UI components
 */

const ItemRenderer = {
    /**
     * Render gallery item
     */
    renderGalleryItem(item) {
        const isFavorite = typeof Favorites !== 'undefined' && Favorites.isFavorite(item.id);
        return `
            <div class="gallery-item" onclick="GalleryManager.viewImage('${item.image}')">
                <img src="${item.image}" alt="${item.description}" loading="lazy">
                <div class="gallery-item-overlay">
                    <div>${this.escapeHtml(item.description)}</div>
                    <div style="font-size: 10px; margin-top: 5px;">
                        ${new Date(item.timestamp).toLocaleDateString()}
                    </div>
                    <button class="favorite-btn ${isFavorite ? 'active' : ''}" 
                            onclick="event.stopPropagation(); GalleryManager.toggleFavorite(${item.id})"
                            title="${isFavorite ? 'Quitar de favoritos' : 'Agregar a favoritos'}">
                        ${isFavorite ? '❤️' : '🤍'}
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Render history item
     */
    renderHistoryItem(item) {
        const isFavorite = typeof Favorites !== 'undefined' && Favorites.isFavorite(item.id);
        return `
            <div class="history-item" onclick="HistoryManager.loadItem(${item.id})">
                <img src="${item.resultImage}" alt="Resultado" loading="lazy">
                <div class="history-info">
                    <h4>${this.escapeHtml(item.clothingDescription || 'Sin descripción')}</h4>
                    <p>${item.characterName ? `Personaje: ${this.escapeHtml(item.characterName)}` : ''}</p>
                    <p>${new Date(item.timestamp).toLocaleString()}</p>
                    <button class="favorite-btn ${isFavorite ? 'active' : ''}" 
                            onclick="event.stopPropagation(); HistoryManager.toggleFavorite(${item.id})"
                            title="${isFavorite ? 'Quitar de favoritos' : 'Agregar a favoritos'}">
                        ${isFavorite ? '❤️' : '🤍'}
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Render list of items
     */
    renderList(items, renderItem, emptyMessage = 'No hay elementos') {
        if (items.length === 0) {
            return this.renderEmptyState(emptyMessage);
        }
        return items.map(renderItem).join('');
    },

    /**
     * Render empty state
     */
    renderEmptyState(message) {
        return `<p style="text-align: center; color: #666; padding: 40px;">${this.escapeHtml(message)}</p>`;
    },

    /**
     * Render loading state
     */
    renderLoading(message = 'Cargando...') {
        return `
            <div class="loading" style="text-align: center; padding: 40px;">
                <div class="spinner"></div>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
    },

    /**
     * Render error state
     */
    renderError(message) {
        return `
            <div class="error-message" style="text-align: center; padding: 40px;">
                <h3>❌ Error</h3>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
    },

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Render statistics card
     */
    renderStatCard(label, value, icon = '') {
        return `
            <div class="stat-item">
                <div class="stat-value">${icon} ${value}</div>
                <div class="stat-label">${this.escapeHtml(label)}</div>
            </div>
        `;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ItemRenderer;
}
