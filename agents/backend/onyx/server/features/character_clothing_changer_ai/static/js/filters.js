/**
 * Filters Module
 * ==============
 * Handles filtering and searching functionality
 */

const Filters = {
    /**
     * Filter gallery items
     */
    filterGallery(searchTerm) {
        const items = GalleryManager.items;
        if (!searchTerm) {
            GalleryManager.load();
            return;
        }

        const filtered = SearchFilter.filter(items, searchTerm, [
            'description',
            'timestamp'
        ]);

        const galleryContent = document.getElementById('galleryContent');
        if (filtered.length === 0) {
            galleryContent.innerHTML = ItemRenderer.renderEmptyState('No se encontraron resultados.');
            return;
        }

        galleryContent.innerHTML = ItemRenderer.renderList(
            filtered,
            (item) => ItemRenderer.renderGalleryItem(item),
            ''
        );
    },

    /**
     * Filter history items
     */
    filterHistory(searchTerm) {
        const items = HistoryManager.items;
        if (!searchTerm) {
            HistoryManager.load();
            return;
        }

        const filtered = SearchFilter.filter(items, searchTerm, [
            'clothingDescription',
            'characterName',
            'timestamp'
        ]);

        const historyContent = document.getElementById('historyContent');
        if (filtered.length === 0) {
            historyContent.innerHTML = ItemRenderer.renderEmptyState('No se encontraron resultados.');
            return;
        }

        historyContent.innerHTML = ItemRenderer.renderList(
            filtered,
            (item) => ItemRenderer.renderHistoryItem(item),
            ''
        );
    },

    /**
     * Sort gallery by date
     */
    sortGallery(order = 'desc') {
        const sorted = SearchFilter.sortByDate(GalleryManager.items, order);
        GalleryManager.items = sorted;
        GalleryManager.load();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Filters;
}


