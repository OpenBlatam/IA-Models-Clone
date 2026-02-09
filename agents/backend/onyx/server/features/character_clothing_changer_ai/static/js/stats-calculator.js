/**
 * Stats Calculator Module
 * =======================
 * Calculates statistics and metrics
 */

const StatsCalculator = {
    /**
     * Calculate frequency of values in array
     */
    calculateFrequency(items, field) {
        const frequency = {};
        items.forEach(item => {
            const value = item[field] || 'N/A';
            frequency[value] = (frequency[value] || 0) + 1;
        });
        return frequency;
    },

    /**
     * Get most frequent value
     */
    getMostFrequent(frequency) {
        let maxCount = 0;
        let mostFrequent = 'N/A';
        
        for (const [value, count] of Object.entries(frequency)) {
            if (count > maxCount) {
                maxCount = count;
                mostFrequent = value;
            }
        }
        
        return { value: mostFrequent, count: maxCount };
    },

    /**
     * Calculate average of numeric field
     */
    calculateAverage(items, field) {
        if (!items || items.length === 0) return 0;
        
        const values = items
            .map(item => item[field])
            .filter(val => typeof val === 'number' && !isNaN(val));
        
        if (values.length === 0) return 0;
        
        const sum = values.reduce((acc, val) => acc + val, 0);
        return sum / values.length;
    },

    /**
     * Get min/max values
     */
    getMinMax(items, field) {
        if (!items || items.length === 0) {
            return { min: null, max: null };
        }
        
        const values = items
            .map(item => item[field])
            .filter(val => typeof val === 'number' && !isNaN(val));
        
        if (values.length === 0) {
            return { min: null, max: null };
        }
        
        return {
            min: Math.min(...values),
            max: Math.max(...values)
        };
    },

    /**
     * Count items by condition
     */
    countBy(items, conditionFn) {
        return items.filter(conditionFn).length;
    },

    /**
     * Get latest item
     */
    getLatest(items, timestampField = 'timestamp') {
        if (!items || items.length === 0) return null;
        
        return items.reduce((latest, item) => {
            const itemTime = new Date(item[timestampField] || 0).getTime();
            const latestTime = new Date(latest[timestampField] || 0).getTime();
            return itemTime > latestTime ? item : latest;
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StatsCalculator;
}

