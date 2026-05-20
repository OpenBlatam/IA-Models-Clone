/**
 * Form Module
 * ===========
 * Handles form interactions and file uploads
 */

const Form = {
    currentImage: null,

    /**
     * Initialize form handlers
     */
    init() {
        this.setupFileUpload();
        this.setupDragAndDrop();
        this.setupFormSubmit();
    },

    /**
     * Setup file upload handler
     */
    setupFileUpload() {
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const fileLabel = document.getElementById('fileLabel');
        const imageAnalysis = document.getElementById('imageAnalysis');

        imageInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            // Validate file
            if (typeof FileHandler !== 'undefined') {
                const validation = FileHandler.validateFile(file);
                if (!validation.valid) {
                    if (typeof Notifications !== 'undefined') {
                        Notifications.error(validation.errors.join(', '));
                    }
                    return;
                }
            }
            
            try {
                const dataURL = typeof FileHandler !== 'undefined'
                    ? await FileHandler.setupPreview(file, preview, fileLabel)
                    : await this.readFileAsDataURL(file);
                
                this.currentImage = dataURL;
                
                // Update state
                if (typeof StateManager !== 'undefined') {
                    StateManager.set('currentImage', dataURL);
                }
                
                // Emit event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('form:image:selected', dataURL);
                }
                
                // Analyze image
                if (typeof ImageAnalyzer !== 'undefined') {
                    ImageAnalyzer.analyze(dataURL);
                }
                if (imageAnalysis) {
                    imageAnalysis.style.display = 'block';
                }
            } catch (error) {
                if (typeof ErrorHandler !== 'undefined') {
                    ErrorHandler.handle(error, { context: 'file upload' });
                } else {
                    console.error('Error reading file:', error);
                }
            }
        });
    },
    
    /**
     * Read file as data URL (fallback)
     */
    readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    /**
     * Setup drag and drop
     */
    setupDragAndDrop() {
        const fileUpload = document.querySelector('.file-upload');
        const fileInput = document.getElementById('imageInput');
        
        if (!fileUpload || !fileInput) return;
        
        if (typeof DragDropHandler !== 'undefined') {
            DragDropHandler.setupForFileInput(fileUpload, fileInput);
        } else {
            // Fallback implementation
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                fileUpload.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
            });
            
            fileUpload.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }, false);
        }
    },

    /**
     * Setup form submission
     */
    setupFormSubmit() {
        const form = document.getElementById('clothingForm');
        const submitBtn = document.getElementById('submitBtn');
        const results = document.getElementById('results');
        const resultContent = document.getElementById('resultContent');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Prepare submission
            const formStartTime = typeof FormSubmissionHandler !== 'undefined'
                ? FormSubmissionHandler.prepareSubmission(submitBtn, results, resultContent)
                : Date.now();

            try {
                // Validate form before submission
                if (typeof FormDataBuilder !== 'undefined') {
                    const validation = FormDataBuilder.validate();
                    if (!validation.valid) {
                        const errorMessage = typeof I18n !== 'undefined' 
                            ? validation.errors.map(e => I18n.t('error.validation') + ': ' + e).join(', ')
                            : validation.errors.join(', ');
                        throw new Error(errorMessage);
                    }
                }
                
                // Build form data
                let formData = typeof FormDataBuilder !== 'undefined' 
                    ? FormDataBuilder.build() 
                    : this.buildFormData();
                
                // Execute before_submit hook
                if (typeof PluginManager !== 'undefined') {
                    formData = PluginManager.executeHook('form:before_submit', formData);
                } else if (typeof PluginSystem !== 'undefined') {
                    formData = PluginSystem.executeHook('form:before_submit', formData);
                }
                
                const response = await API.changeClothing(formData);

                if (!response.success) {
                    throw new Error(response.error);
                }

                let data = response.data;
                
                // Execute after_submit hook
                if (typeof PluginManager !== 'undefined') {
                    data = PluginManager.executeHook('form:after_submit', data);
                } else if (typeof PluginSystem !== 'undefined') {
                    data = PluginSystem.executeHook('form:after_submit', data);
                }
                
                // Handle success
                if (typeof FormSubmissionHandler !== 'undefined') {
                    FormSubmissionHandler.handleSuccess(data, this.currentImage, submitBtn, resultContent);
                } else {
                    // Fallback success handling
                    if (typeof HistoryManager !== 'undefined') HistoryManager.add(data);
                    if (typeof GalleryManager !== 'undefined') GalleryManager.add(data);
                    if (typeof UI !== 'undefined') {
                        resultContent.innerHTML = UI.showResult(data);
                        UI.switchTab('result');
                    }
                    if (typeof Notifications !== 'undefined') {
                        Notifications.success('¡Ropa cambiada exitosamente!');
                    }
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Cambiar Ropa';
                }
                
            } catch (error) {
                // Handle error
                if (typeof FormSubmissionHandler !== 'undefined') {
                    FormSubmissionHandler.handleError(error, submitBtn, resultContent);
                } else {
                    // Fallback error handling
                    if (typeof UI !== 'undefined') {
                        resultContent.innerHTML = UI.showError(error.message || error.toString());
                    }
                    if (typeof Notifications !== 'undefined') {
                        Notifications.error('Error al procesar la solicitud');
                    }
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Cambiar Ropa';
                }
            }
        });
    },

};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Form;
}
