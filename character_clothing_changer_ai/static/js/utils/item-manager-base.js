/**
 * Item Manager Base Module
 * =========================
 * Base class for managing collections of items (gallery, history, etc.)
 */

const ItemManagerBase = {
    /**
     * Create item from data
     */
    createItem(data, options = {}) {
        const {
            idGenerator = () => Date.now(),
            imageFields = ['image_base64', 'image_url', 'result_image'],
            additionalFields = {}
        } = options;
        
        const resultImage = imageFields.find(field => data[field]) 
            ? imageFields.find(field => data[field]) 
            : null;
        
        const imageValue = resultImage ? data[resultImage] : null;
        
        return {
            id: idGenerator(),
            timestamp: new Date().toISOString(),
            ...this.extractImageFields(data, imageFields),
            clothing_description: data.clothing_description || this.getFormValue('clothingDescription') || '',
            character_name: data.character_name || this.getFormValue('characterName') || '',
            prompt_used: data.prompt_used,
            negative_prompt_used: data.negative_prompt_used,
            quality_metrics: data.quality_metrics,
            saved_path: data.saved_path,
            ...additionalFields
        };
    },

    /**
     * Extract image fields from data
     */
    extractImageFields(data, imageFields) {
        const result = {};
        imageFields.forEach(field => {
            if (data[field]) {
                result[field] = data[field];
            }
        });
        return result;
    },

    /**
     * Get form value
     */
    getFormValue(fieldId) {
        const element = document.getElementById(fieldId);
        return element ? element.value : null;
    },

    /**
     * Add item to collection
     */
    addItem(item, collection, storageKey, maxItems, eventName) {
        // Validate item has image
        const hasImage = item.image_base64 || item.image_url || item.result_image;
        if (!hasImage) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Cannot add item: no image`, { item });
            }
            return;
        }

        collection.unshift(item);
        
        // Enforce max items limit
        if (maxItems && collection.length > maxItems) {
            collection.pop();
        }
        
        // Save to storage
        if (typeof Storage !== 'undefined') {
            Storage.set(storageKey, collection);
        }
        
        // Emit event
        if (typeof EventBus !== 'undefined' && eventName) {
            EventBus.emit(eventName, item);
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Item added`, { 
                characterName: item.character_name,
                id: item.id 
            });
        }
    },

    /**
     * Load items display
     */
    loadItems(items, containerId, renderFn, emptyMessage, eventName) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        if (items.length === 0) {
            container.innerHTML = typeof ItemRenderer !== 'undefined' 
                ? ItemRenderer.renderEmptyState(emptyMessage)
                : `<p>${emptyMessage}</p>`;
            return;
        }
        
        // Render items
        if (typeof ItemRenderer !== 'undefined' && renderFn) {
            container.innerHTML = items.map((item, index) => 
                renderFn(item, index)
            ).join('');
        } else {
            // Fallback rendering
            container.innerHTML = items.map((item, index) => `
                <div class="item">
                    <img src="${item.image_base64 || item.image_url || item.result_image}" alt="Item ${index}">
                </div>
            `).join('');
        }
        
        // Emit loaded event
        if (typeof EventBus !== 'undefined' && eventName) {
            EventBus.emit(eventName, items);
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.debug('Items loaded', { itemCount: items.length });
        }
    },

    /**
     * Find item by ID
     */
    findItemById(items, id) {
        return items.find(item => item.id === id);
    },

    /**
     * Get image URL from item
     */
    getImageUrl(item) {
        return item.image_base64 || item.image_url || item.result_image || null;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ItemManagerBase;
}

