/**
 * File Downloader Module
 * ======================
 * Handles file downloads (images, JSON, etc.)
 */

const FileDownloader = {
    /**
     * Download image
     */
    downloadImage(imageSrc, filename = null) {
        try {
            const link = document.createElement('a');
            link.href = imageSrc;
            link.download = filename || `clothing_changed_${Date.now()}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Imagen descargada exitosamente');
            }
            return true;
        } catch (error) {
            console.error('Error downloading image:', error);
            if (typeof Notifications !== 'undefined') {
                Notifications.error('Error al descargar la imagen');
            }
            return false;
        }
    },

    /**
     * Download JSON data
     */
    downloadJSON(data, filename = null) {
        try {
            const blob = new Blob([JSON.stringify(data, null, 2)], { 
                type: 'application/json' 
            });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename || `data_${Date.now()}.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
            
            if (typeof Notifications !== 'undefined') {
                Notifications.success('Archivo descargado exitosamente');
            }
            return true;
        } catch (error) {
            console.error('Error downloading JSON:', error);
            if (typeof Notifications !== 'undefined') {
                Notifications.error('Error al descargar el archivo');
            }
            return false;
        }
    },

    /**
     * Download text file
     */
    downloadText(text, filename, mimeType = 'text/plain') {
        try {
            const blob = new Blob([text], { type: mimeType });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
            
            return true;
        } catch (error) {
            console.error('Error downloading text:', error);
            return false;
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FileDownloader;
}

