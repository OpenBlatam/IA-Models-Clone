/**
 * Config Exporter Module
 * =====================
 * Handles configuration export/import
 */

const ConfigExporter = {
    /**
     * Export current form configuration
     */
    export() {
        try {
            const config = this.getCurrentConfig();
            const filename = `clothing_config_${Date.now()}.json`;
            return FileDownloader.downloadJSON(config, filename);
        } catch (error) {
            console.error('Error exporting config:', error);
            if (typeof Notifications !== 'undefined') {
                Notifications.error('Error al exportar la configuración');
            }
            return false;
        }
    },

    /**
     * Get current form configuration
     */
    getCurrentConfig() {
        return {
            clothingDescription: this.getInputValue('clothingDescription'),
            characterName: this.getInputValue('characterName'),
            prompt: this.getInputValue('prompt'),
            negativePrompt: this.getInputValue('negativePrompt'),
            numSteps: this.getInputValue('numSteps', '50'),
            guidanceScale: this.getInputValue('guidanceScale', '7.5'),
            strength: this.getInputValue('strength', '0.8'),
            saveTensor: this.getInputValue('saveTensor', 'false'),
            timestamp: new Date().toISOString()
        };
    },

    /**
     * Import configuration
     */
    import(config) {
        try {
            if (config.clothingDescription) {
                this.setInputValue('clothingDescription', config.clothingDescription);
            }
            if (config.characterName) {
                this.setInputValue('characterName', config.characterName);
            }
            if (config.prompt) {
                this.setInputValue('prompt', config.prompt);
            }
            if (config.negativePrompt) {
                this.setInputValue('negativePrompt', config.negativePrompt);
            }
            if (config.numSteps) {
                this.setInputValue('numSteps', config.numSteps);
            }
            if (config.guidanceScale) {
                this.setInputValue('guidanceScale', config.guidanceScale);
            }
            if (config.strength) {
                this.setInputValue('strength', config.strength);
            }
            if (config.saveTensor) {
                this.setInputValue('saveTensor', config.saveTensor);
            }
            
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Configuración importada exitosamente');
            }
            return true;
        } catch (error) {
            console.error('Error importing config:', error);
            if (typeof Notifications !== 'undefined') {
                Notifications.error('Error al importar la configuración');
            }
            return false;
        }
    },

    /**
     * Get input value by ID
     */
    getInputValue(id, defaultValue = '') {
        const element = document.getElementById(id);
        return element ? element.value : defaultValue;
    },

    /**
     * Set input value by ID
     */
    setInputValue(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.value = value;
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConfigExporter;
}

