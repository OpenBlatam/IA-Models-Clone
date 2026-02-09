/**
 * History Module
 * ==============
 * Handles history display and management
 */

const HistoryManager = {
    items: [],

    /**
     * Initialize history
     */
    init() {
        this.items = typeof Storage !== 'undefined' ? Storage.getHistory() : [];
        this.load();
    },

    /**
     * Add item to history
     */
    add(data) {
        if (typeof ItemManagerBase === 'undefined') {
            // Fallback implementation
            this.addFallback(data);
            return;
        }
        
        const item = ItemManagerBase.createItem(data, {
            imageFields: ['image_base64', 'image_url', 'result_image'],
            additionalFields: {
                originalImage: typeof Form !== 'undefined' ? Form.currentImage : null,
                resultImage: data.image_base64 || data.image_url || data.result_image
            }
        });
        
        ItemManagerBase.addItem(
            item,
            this.items,
            typeof CONFIG !== 'undefined' ? CONFIG.STORAGE_KEYS.HISTORY : 'history',
            typeof CONFIG !== 'undefined' ? CONFIG.LIMITS.MAX_HISTORY : 100,
            'history:item:added'
        );
    },

    /**
     * Fallback add implementation
     */
    addFallback(data) {
        const resultImage = data.image_base64 || data.image_url || data.result_image;
        
        const item = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            originalImage: typeof Form !== 'undefined' ? Form.currentImage : null,
            resultImage: resultImage,
            image_base64: data.image_base64,
            image_url: data.image_url,
            clothing_description: data.clothing_description || '',
            character_name: data.character_name || ''
        };

        this.items.unshift(item);
        if (typeof Storage !== 'undefined') {
            Storage.saveHistory(this.items);
        }
    },

    /**
     * Load history display
     */
    load() {
        if (typeof ItemManagerBase !== 'undefined') {
            ItemManagerBase.loadItems(
                this.items,
                'historyContent',
                (item, index) => typeof ItemRenderer !== 'undefined' 
                    ? ItemRenderer.renderHistoryItem(item, index)
                    : this.renderItemFallback(item, index),
                'No hay historial aún.',
                'history:loaded'
            );
        } else {
            // Fallback implementation
            this.loadFallback();
        }
    },

    /**
     * Fallback load implementation
     */
    loadFallback() {
        const historyContent = document.getElementById('historyContent');
        if (!historyContent) return;
        
        if (this.items.length === 0) {
            historyContent.innerHTML = '<p>No hay historial aún.</p>';
            return;
        }
        
        historyContent.innerHTML = this.items.map((item, index) => 
            this.renderItemFallback(item, index)
        ).join('');
    },

    /**
     * Render item fallback
     */
    renderItemFallback(item, index) {
        return `
            <div class="history-item">
                <img src="${item.resultImage || item.image_base64 || item.image_url}" alt="History item ${index}">
                <div>${item.clothing_description || 'Sin descripción'}</div>
            </div>
        `;
    },

    /**
     * Load history item
     */
    loadItem(id) {
        const item = typeof ItemManagerBase !== 'undefined'
            ? ItemManagerBase.findItemById(this.items, id)
            : this.items.find(h => h.id === id);
            
        if (!item) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('History item not found', { id });
            }
            return;
        }

        // Update form with item data
        if (typeof Form !== 'undefined') {
            Form.currentImage = item.originalImage;
        }
        
        if (typeof StateManager !== 'undefined') {
            StateManager.set('currentImage', item.originalImage);
            StateManager.set('currentResult', { image_base64: item.resultImage });
        } else if (typeof AppState !== 'undefined') {
            AppState.currentResult = { image_base64: item.resultImage };
        }
        
        const descriptionField = document.getElementById('clothingDescription');
        if (descriptionField) {
            descriptionField.value = item.clothing_description || item.clothingDescription || '';
        }
        
        const nameField = document.getElementById('characterName');
        if (nameField && (item.character_name || item.characterName)) {
            nameField.value = item.character_name || item.characterName;
        }
        
        if (typeof Comparison !== 'undefined') {
            Comparison.update(item.originalImage, item.resultImage);
        }
        
        if (typeof UI !== 'undefined') {
            UI.switchTab('comparison');
        }
        
        if (typeof Notifications !== 'undefined') {
            Notifications.info('Item cargado desde historial');
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('History item loaded', { id, characterName: item.character_name || item.characterName });
        }
    },

    /**
     * Toggle favorite status
     */
    toggleFavorite(itemId) {
        Favorites.toggle(itemId);
        this.load();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HistoryManager;
}
