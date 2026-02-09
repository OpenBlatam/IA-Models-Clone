/**
 * Array Utils Module
 * ==================
 * Utilities for array manipulation and operations
 */

const ArrayUtils = {
    /**
     * Chunk array into smaller arrays
     */
    chunk(array, size) {
        if (!Array.isArray(array) || size <= 0) {
            return [];
        }
        
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    },
    
    /**
     * Remove duplicates
     */
    unique(array, key = null) {
        if (!Array.isArray(array)) {
            return [];
        }
        
        if (key) {
            const seen = new Set();
            return array.filter(item => {
                const value = typeof key === 'function' ? key(item) : item[key];
                if (seen.has(value)) {
                    return false;
                }
                seen.add(value);
                return true;
            });
        }
        
        return [...new Set(array)];
    },
    
    /**
     * Flatten nested array
     */
    flatten(array, depth = Infinity) {
        if (!Array.isArray(array)) {
            return [];
        }
        
        return array.flat(depth);
    },
    
    /**
     * Group by key
     */
    groupBy(array, key) {
        if (!Array.isArray(array)) {
            return {};
        }
        
        const getKey = typeof key === 'function' ? key : item => item[key];
        
        return array.reduce((groups, item) => {
            const groupKey = getKey(item);
            if (!groups[groupKey]) {
                groups[groupKey] = [];
            }
            groups[groupKey].push(item);
            return groups;
        }, {});
    },
    
    /**
     * Sort by key
     */
    sortBy(array, key, order = 'asc') {
        if (!Array.isArray(array)) {
            return [];
        }
        
        const getValue = typeof key === 'function' ? key : item => item[key];
        const sorted = [...array].sort((a, b) => {
            const aVal = getValue(a);
            const bVal = getValue(b);
            
            if (aVal < bVal) return order === 'asc' ? -1 : 1;
            if (aVal > bVal) return order === 'asc' ? 1 : -1;
            return 0;
        });
        
        return sorted;
    },
    
    /**
     * Shuffle array
     */
    shuffle(array) {
        if (!Array.isArray(array)) {
            return [];
        }
        
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    },
    
    /**
     * Get random item
     */
    random(array) {
        if (!Array.isArray(array) || array.length === 0) {
            return null;
        }
        
        return array[Math.floor(Math.random() * array.length)];
    },
    
    /**
     * Get random items
     */
    randomItems(array, count) {
        if (!Array.isArray(array) || count <= 0) {
            return [];
        }
        
        const shuffled = this.shuffle(array);
        return shuffled.slice(0, Math.min(count, array.length));
    },
    
    /**
     * Intersection of arrays
     */
    intersection(array1, array2) {
        if (!Array.isArray(array1) || !Array.isArray(array2)) {
            return [];
        }
        
        return array1.filter(item => array2.includes(item));
    },
    
    /**
     * Difference of arrays
     */
    difference(array1, array2) {
        if (!Array.isArray(array1)) {
            return [];
        }
        if (!Array.isArray(array2)) {
            return array1;
        }
        
        return array1.filter(item => !array2.includes(item));
    },
    
    /**
     * Union of arrays
     */
    union(array1, array2) {
        if (!Array.isArray(array1) && !Array.isArray(array2)) {
            return [];
        }
        if (!Array.isArray(array1)) {
            return array2;
        }
        if (!Array.isArray(array2)) {
            return array1;
        }
        
        return this.unique([...array1, ...array2]);
    },
    
    /**
     * Partition array
     */
    partition(array, predicate) {
        if (!Array.isArray(array)) {
            return [[], []];
        }
        
        const truthy = [];
        const falsy = [];
        
        array.forEach(item => {
            if (predicate(item)) {
                truthy.push(item);
            } else {
                falsy.push(item);
            }
        });
        
        return [truthy, falsy];
    },
    
    /**
     * Find by key-value
     */
    findBy(array, key, value) {
        if (!Array.isArray(array)) {
            return null;
        }
        
        return array.find(item => item[key] === value);
    },
    
    /**
     * Find all by key-value
     */
    findAllBy(array, key, value) {
        if (!Array.isArray(array)) {
            return [];
        }
        
        return array.filter(item => item[key] === value);
    },
    
    /**
     * Remove item
     */
    remove(array, item) {
        if (!Array.isArray(array)) {
            return [];
        }
        
        const index = array.indexOf(item);
        if (index > -1) {
            array.splice(index, 1);
        }
        return array;
    },
    
    /**
     * Remove by key-value
     */
    removeBy(array, key, value) {
        if (!Array.isArray(array)) {
            return [];
        }
        
        return array.filter(item => item[key] !== value);
    },
    
    /**
     * Sum array
     */
    sum(array, key = null) {
        if (!Array.isArray(array)) {
            return 0;
        }
        
        const getValue = key ? (item => item[key]) : (item => item);
        return array.reduce((sum, item) => {
            const value = getValue(item);
            return sum + (typeof value === 'number' ? value : 0);
        }, 0);
    },
    
    /**
     * Average array
     */
    average(array, key = null) {
        if (!Array.isArray(array) || array.length === 0) {
            return 0;
        }
        
        return this.sum(array, key) / array.length;
    },
    
    /**
     * Min value
     */
    min(array, key = null) {
        if (!Array.isArray(array) || array.length === 0) {
            return null;
        }
        
        const getValue = key ? (item => item[key]) : (item => item);
        return Math.min(...array.map(getValue));
    },
    
    /**
     * Max value
     */
    max(array, key = null) {
        if (!Array.isArray(array) || array.length === 0) {
            return null;
        }
        
        const getValue = key ? (item => item[key]) : (item => item);
        return Math.max(...array.map(getValue));
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ArrayUtils;
}

