/**
 * Search Filter Module
 * ====================
 * Advanced search and filtering utilities
 */

const SearchFilter = {
    /**
     * Filter items by search term across multiple fields
     */
    filter(items, searchTerm, fields = []) {
        if (!searchTerm || !searchTerm.trim()) {
            return items;
        }
        
        const term = searchTerm.toLowerCase().trim();
        
        return items.filter(item => {
            // If no fields specified, search all string properties
            const searchFields = fields.length > 0 ? fields : Object.keys(item).filter(key => 
                typeof item[key] === 'string' || typeof item[key] === 'number'
            );
            
            return searchFields.some(field => {
                const value = item[field];
                if (value === null || value === undefined) {
                    return false;
                }
                
                // Handle dates
                if (field === 'timestamp' && typeof value === 'string') {
                    try {
                        const date = new Date(value);
                        return date.toLocaleString().toLowerCase().includes(term);
                    } catch {
                        return value.toLowerCase().includes(term);
                    }
                }
                
                // Handle strings and numbers
                return String(value).toLowerCase().includes(term);
            });
        });
    },

    /**
     * Sort items by date
     */
    sortByDate(items, order = 'desc') {
        const sorted = [...items];
        sorted.sort((a, b) => {
            const dateA = new Date(a.timestamp || 0);
            const dateB = new Date(b.timestamp || 0);
            return order === 'desc' ? dateB - dateA : dateA - dateB;
        });
        return sorted;
    },

    /**
     * Sort items by field
     */
    sortBy(items, field, order = 'asc') {
        const sorted = [...items];
        sorted.sort((a, b) => {
            const aVal = a[field];
            const bVal = b[field];
            
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;
            
            if (typeof aVal === 'string' && typeof bVal === 'string') {
                return order === 'asc' 
                    ? aVal.localeCompare(bVal)
                    : bVal.localeCompare(aVal);
            }
            
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return order === 'asc' ? aVal - bVal : bVal - aVal;
            }
            
            return 0;
        });
        return sorted;
    },

    /**
     * Filter and sort combined
     */
    filterAndSort(items, searchTerm, sortField = null, sortOrder = 'desc') {
        let result = items;
        
        if (searchTerm) {
            result = this.filter(result, searchTerm);
        }
        
        if (sortField) {
            if (sortField === 'date' || sortField === 'timestamp') {
                result = this.sortByDate(result, sortOrder);
            } else {
                result = this.sortBy(result, sortField, sortOrder);
            }
        }
        
        return result;
    },

    /**
     * Highlight search term in text
     */
    highlight(text, searchTerm) {
        if (!searchTerm || !text) return text;
        
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SearchFilter;
}
