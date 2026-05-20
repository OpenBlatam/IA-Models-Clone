/**
 * Gallery Module
 * ==============
 * Handles gallery display and management
 */

const GalleryManager = {
    items: [],

    /**
     * Initialize gallery
     */
    init() {
        this.items = typeof Storage !== 'undefined' ? Storage.getGallery() : [];
        this.load();
    },

    /**
     * Add item to gallery
     */
    add(data) {
        if (typeof ItemManagerBase === 'undefined') {
            // Fallback implementation
            this.addFallback(data);
            return;
        }
        
        const item = ItemManagerBase.createItem(data, {
            imageFields: ['image_base64', 'image_url', 'result_image']
        });
        
        ItemManagerBase.addItem(
            item,
            this.items,
            typeof CONFIG !== 'undefined' ? CONFIG.STORAGE_KEYS.GALLERY : 'gallery',
            typeof CONFIG !== 'undefined' ? CONFIG.LIMITS.MAX_GALLERY : 100,
            'gallery:item:added'
        );
    },

    /**
     * Fallback add implementation
     */
    addFallback(data) {
        const resultImage = data.image_base64 || data.image_url || data.result_image;
        if (!resultImage) return;

        const item = {
            id: Date.now(),
            image_base64: resultImage,
            image_url: data.image_url,
            result_image: resultImage,
            clothing_description: data.clothing_description || '',
            character_name: data.character_name || '',
            timestamp: new Date().toISOString()
        };

        this.items.unshift(item);
        if (typeof Storage !== 'undefined') {
            Storage.saveGallery(this.items);
        }
    },

    /**
     * Load gallery display
     */
    load() {
        if (typeof ItemManagerBase !== 'undefined') {
            ItemManagerBase.loadItems(
                this.items,
                'galleryContent',
                (item, index) => typeof ItemRenderer !== 'undefined' 
                    ? ItemRenderer.renderGalleryItem(item, index)
                    : this.renderItemFallback(item, index),
                'No hay imágenes en la galería aún.',
                'gallery:loaded'
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
        const galleryContent = document.getElementById('galleryContent');
        if (!galleryContent) return;
        
        if (this.items.length === 0) {
            galleryContent.innerHTML = '<p>No hay imágenes en la galería aún.</p>';
            return;
        }
        
        galleryContent.innerHTML = this.items.map((item, index) => 
            this.renderItemFallback(item, index)
        ).join('');
    },

    /**
     * Render item fallback
     */
    renderItemFallback(item, index) {
        return `
            <div class="gallery-item">
                <img src="${item.image_base64 || item.image_url || item.result_image}" alt="Gallery item ${index}">
            </div>
        `;
    },

    /**
     * View image in modal
     */
    viewImage(index) {
        const item = this.items[index];
        if (!item) return;
        
        const imageUrl = typeof ItemManagerBase !== 'undefined'
            ? ItemManagerBase.getImageUrl(item)
            : (item.image_base64 || item.image_url);
        
        if (!imageUrl) return;
        
        if (typeof ModalViewer !== 'undefined') {
            ModalViewer.show(imageUrl);
        } else {
            // Fallback: open in new window
            window.open(imageUrl, '_blank');
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
    module.exports = GalleryManager;
}
