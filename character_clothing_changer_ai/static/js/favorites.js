/**
 * Favorites Module
 * ===============
 * Handles favorite items management
 */

const Favorites = {
    items: [],

    /**
     * Initialize favorites
     */
    init() {
        this.items = LocalStorageWrapper.get('favorites', []);
    },

    /**
     * Toggle favorite status
     */
    toggle(itemId) {
        const index = this.items.indexOf(itemId);
        if (index > -1) {
            this.items.splice(index, 1);
            if (typeof Notifications !== 'undefined') {
                Notifications.info('Removido de favoritos');
            }
        } else {
            this.items.push(itemId);
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Agregado a favoritos');
            }
        }
        LocalStorageWrapper.set('favorites', this.items);
        return index === -1;
    },

    /**
     * Check if item is favorite
     */
    isFavorite(itemId) {
        return this.items.includes(itemId);
    },

    /**
     * Get all favorites
     */
    getAll() {
        return this.items;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Favorites;
}


