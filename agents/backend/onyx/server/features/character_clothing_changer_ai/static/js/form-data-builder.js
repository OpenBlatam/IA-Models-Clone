/**
 * Form Data Builder Module
 * ========================
 * Handles form data construction and validation
 */

const FormDataBuilder = {
    /**
     * Build FormData from form inputs
     */
    build() {
        const formData = new FormData();
        const imageFile = document.getElementById('imageInput').files[0];
        
        if (!imageFile) {
            throw new Error('Por favor selecciona una imagen');
        }

        formData.append('image', imageFile);
        formData.append('clothing_description', this.getClothingDescription());
        
        const characterName = this.getCharacterName();
        if (characterName) {
            formData.append('character_name', characterName);
        }

        const prompt = this.getPrompt();
        if (prompt) {
            formData.append('prompt', prompt);
        }

        const negativePrompt = this.getNegativePrompt();
        if (negativePrompt) {
            formData.append('negative_prompt', negativePrompt);
        }

        formData.append('num_inference_steps', this.getNumSteps());
        formData.append('guidance_scale', this.getGuidanceScale());
        formData.append('strength', this.getStrength());
        formData.append('save_tensor', this.getSaveTensor());

        return formData;
    },

    /**
     * Validate form data using Validator module
     */
    validate() {
        const imageFile = document.getElementById('imageInput').files[0];
        const description = this.getClothingDescription();
        const numSteps = this.getNumSteps();
        const guidanceScale = this.getGuidanceScale();
        const strength = this.getStrength();
        
        const errors = [];

        // Validate image using Validator
        if (typeof Validator !== 'undefined') {
            const imageValidation = Validator.imageFile(imageFile);
            if (!imageValidation.valid) {
                errors.push(imageValidation.error);
            }
        } else {
            // Fallback validation
            if (!imageFile) {
                errors.push('Debes seleccionar una imagen');
            } else {
                const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
                if (!validTypes.includes(imageFile.type)) {
                    errors.push('El archivo debe ser PNG, JPG o JPEG');
                }
                const maxSize = 10 * 1024 * 1024;
                if (imageFile.size > maxSize) {
                    errors.push('El archivo es demasiado grande (máximo 10MB)');
                }
            }
        }

        // Validate description
        if (typeof Validator !== 'undefined') {
            const requiredValidation = Validator.required(description);
            if (!requiredValidation.valid) {
                errors.push(requiredValidation.error);
            } else {
                const lengthValidation = Validator.textLength(description, 3, 500);
                if (!lengthValidation.valid) {
                    errors.push(lengthValidation.error);
                }
            }
        } else {
            if (!description || description.trim().length === 0) {
                errors.push('Debes describir la ropa que quieres');
            } else if (description.length < 3) {
                errors.push('La descripción debe tener al menos 3 caracteres');
            }
        }

        // Validate num steps
        if (typeof Validator !== 'undefined') {
            const stepsValidation = Validator.numberRange(numSteps, 1, 100);
            if (!stepsValidation.valid) {
                errors.push(stepsValidation.error);
            }
        } else {
            if (numSteps < 1 || numSteps > 100) {
                errors.push('Los pasos de inferencia deben estar entre 1 y 100');
            }
        }

        // Validate guidance scale
        if (typeof Validator !== 'undefined') {
            const guidanceValidation = Validator.numberRange(guidanceScale, 1, 20);
            if (!guidanceValidation.valid) {
                errors.push(guidanceValidation.error);
            }
        } else {
            if (guidanceScale < 1 || guidanceScale > 20) {
                errors.push('El guidance scale debe estar entre 1 y 20');
            }
        }

        // Validate strength
        if (typeof Validator !== 'undefined') {
            const strengthValidation = Validator.numberRange(strength, 0, 1);
            if (!strengthValidation.valid) {
                errors.push(strengthValidation.error);
            }
        } else {
            if (strength < 0 || strength > 1) {
                errors.push('La fuerza de inpainting debe estar entre 0 y 1');
            }
        }

        if (errors.length > 0) {
            const errorMessage = errors.join('\n');
            throw new Error(errorMessage);
        }
    },

    /**
     * Get form values
     */
    getClothingDescription() {
        return document.getElementById('clothingDescription').value.trim();
    },

    getCharacterName() {
        return document.getElementById('characterName').value.trim();
    },

    getPrompt() {
        return document.getElementById('prompt').value.trim();
    },

    getNegativePrompt() {
        return document.getElementById('negativePrompt').value.trim();
    },

    getNumSteps() {
        const value = parseInt(document.getElementById('numSteps').value);
        return isNaN(value) ? CONFIG.DEFAULT_VALUES.NUM_STEPS : value;
    },

    getGuidanceScale() {
        const value = parseFloat(document.getElementById('guidanceScale').value);
        return isNaN(value) ? CONFIG.DEFAULT_VALUES.GUIDANCE_SCALE : value;
    },

    getStrength() {
        const value = parseFloat(document.getElementById('strength').value);
        return isNaN(value) ? CONFIG.DEFAULT_VALUES.STRENGTH : value;
    },

    getSaveTensor() {
        return document.getElementById('saveTensor').value;
    },

    /**
     * Get form data as object (for debugging/export)
     */
    getFormDataObject() {
        return {
            clothingDescription: this.getClothingDescription(),
            characterName: this.getCharacterName(),
            prompt: this.getPrompt(),
            negativePrompt: this.getNegativePrompt(),
            numSteps: this.getNumSteps(),
            guidanceScale: this.getGuidanceScale(),
            strength: this.getStrength(),
            saveTensor: this.getSaveTensor()
        };
    },

    /**
     * Reset form to default values
     */
    reset() {
        document.getElementById('clothingDescription').value = '';
        document.getElementById('characterName').value = '';
        document.getElementById('prompt').value = '';
        document.getElementById('negativePrompt').value = '';
        document.getElementById('numSteps').value = CONFIG.DEFAULT_VALUES.NUM_STEPS;
        document.getElementById('guidanceScale').value = CONFIG.DEFAULT_VALUES.GUIDANCE_SCALE;
        document.getElementById('strength').value = CONFIG.DEFAULT_VALUES.STRENGTH;
        document.getElementById('saveTensor').value = CONFIG.DEFAULT_VALUES.SAVE_TENSOR;
        
        const imageInput = document.getElementById('imageInput');
        if (imageInput) {
            imageInput.value = '';
        }
        
        const preview = document.getElementById('preview');
        if (preview) {
            preview.style.display = 'none';
            preview.src = '';
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FormDataBuilder;
}
