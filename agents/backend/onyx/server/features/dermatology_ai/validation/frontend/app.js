// Configuration
const API_BASE_URL = 'http://localhost:8006';
const API_ENDPOINTS = {
    analyze: `${API_BASE_URL}/dermatology/analyze-image`,
    recommendations: `${API_BASE_URL}/dermatology/get-recommendations`
};

// DOM Elements
const imageInput = document.getElementById('image-input');
const uploadBox = document.getElementById('upload-box');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const status = document.getElementById('status');
const statusText = document.getElementById('status-text');

// State
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkBackendConnection();
    setupEventListeners();
});

// Check Backend Connection
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            showStatus('✅ Backend conectado correctamente', 'success');
            console.log('✅ Backend conectado correctamente');
        } else {
            showStatus('⚠️ Backend responde pero con errores', 'error');
            console.warn('⚠️ Backend responde pero con errores');
        }
    } catch (error) {
        showStatus('❌ No se puede conectar al backend. Asegúrate que esté corriendo en el puerto 8006', 'error');
        console.error('❌ Error conectando al backend:', error);
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Upload box click
    uploadBox.addEventListener('click', () => {
        imageInput.click();
    });

    // File input change
    imageInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);

    // Analyze button
    analyzeBtn.addEventListener('click', handleAnalyze);
}

// File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showError('Por favor, sube solo archivos de imagen (JPG, PNG, etc.)');
    }
}

// Process File
function processFile(file) {
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('El archivo es demasiado grande. Máximo 10MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        uploadBox.querySelector('.upload-content').classList.add('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Analyze Image
async function handleAnalyze() {
    if (!selectedFile) {
        showError('Por favor, selecciona una imagen primero');
        return;
    }

    // Hide previous results/errors
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-text').classList.add('hidden');
    analyzeBtn.querySelector('.btn-loader').classList.remove('hidden');

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('enhance', document.getElementById('enhance').checked);
        formData.append('use_advanced', document.getElementById('advanced').checked);
        formData.append('use_cache', 'true');

        // Make API call
        const response = await fetch(API_ENDPOINTS.analyze, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `Error ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.success && data.analysis) {
            displayResults(data.analysis);
        } else {
            throw new Error(data.message || 'Error en el análisis');
        }

    } catch (error) {
        console.error('Error analizando imagen:', error);
        showError(error.message || 'Error al analizar la imagen. Por favor, intenta de nuevo.');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').classList.remove('hidden');
        analyzeBtn.querySelector('.btn-loader').classList.add('hidden');
    }
}

// Display Results
function displayResults(analysis) {
    // Show results section
    resultsSection.classList.remove('hidden');

    // Overall score
    const overallScore = Math.round(analysis.quality_scores?.overall_score || 0);
    document.getElementById('overall-score').textContent = overallScore;

    // Detailed metrics
    const metrics = {
        'texture': analysis.quality_scores?.texture_score || 0,
        'hydration': analysis.quality_scores?.hydration_score || 0,
        'elasticity': analysis.quality_scores?.elasticity_score || 0,
        'pigmentation': analysis.quality_scores?.pigmentation_score || 0,
        'pore': analysis.quality_scores?.pore_size_score || 0,
        'wrinkles': analysis.quality_scores?.wrinkles_score || 0
    };

    // Update metric cards
    Object.keys(metrics).forEach(key => {
        const score = Math.round(metrics[key]);
        document.getElementById(`${key}-score`).textContent = score;
        document.getElementById(`${key}-fill`).style.width = `${score}%`;
    });

    // Skin type
    const skinType = analysis.skin_type || 'no detectado';
    document.getElementById('skin-type').textContent = skinType;

    // Conditions
    const conditions = analysis.conditions || [];
    const conditionsSection = document.getElementById('conditions-section');
    const conditionsList = document.getElementById('conditions-list');

    if (conditions.length > 0) {
        conditionsSection.classList.remove('hidden');
        conditionsList.innerHTML = conditions.map(condition => `
            <div class="condition-item">
                <span class="condition-name">${condition.name || 'Condición'}</span>
                <span class="condition-severity">${condition.severity || 'moderada'}</span>
            </div>
        `).join('');
    } else {
        conditionsSection.classList.add('hidden');
    }

    // Recommendations
    const recommendations = analysis.recommendations_priority || [];
    const recommendationsSection = document.getElementById('recommendations-section');
    const recommendationsList = document.getElementById('recommendations-list');

    if (recommendations.length > 0) {
        recommendationsSection.classList.remove('hidden');
        recommendationsList.innerHTML = recommendations.map(rec => `
            <div class="recommendation-item">
                <strong>${formatRecommendation(rec)}</strong>
                <p>${getRecommendationDescription(rec)}</p>
            </div>
        `).join('');
    } else {
        recommendationsSection.classList.add('hidden');
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Format Recommendation
function formatRecommendation(rec) {
    const names = {
        'hydration': '💧 Hidratación',
        'texture': '✨ Textura',
        'pore_care': '🔍 Cuidado de Poros',
        'pigmentation': '🎨 Pigmentación',
        'wrinkles': '👵 Arrugas',
        'redness': '🔴 Enrojecimiento',
        'dark_spots': '🌑 Manchas Oscuras'
    };
    return names[rec] || rec;
}

// Get Recommendation Description
function getRecommendationDescription(rec) {
    const descriptions = {
        'hydration': 'Tu piel necesita más hidratación. Usa productos hidratantes y bebe más agua.',
        'texture': 'Mejora la textura de tu piel con exfoliantes suaves y productos nutritivos.',
        'pore_care': 'Cuida tus poros con limpiadores suaves y productos no comedogénicos.',
        'pigmentation': 'Trabaja en la uniformidad de tu pigmentación con productos adecuados.',
        'wrinkles': 'Previene y reduce arrugas con productos anti-edad y protección solar.',
        'redness': 'Reduce el enrojecimiento con productos calmantes y evita irritantes.',
        'dark_spots': 'Trata las manchas oscuras con productos despigmentantes y protección solar.'
    };
    return descriptions[rec] || 'Sigue una rutina de cuidado de piel adecuada.';
}

// Show Error
function showError(message) {
    errorSection.classList.remove('hidden');
    document.getElementById('error-message').textContent = message;
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show Status
function showStatus(message, type = 'info') {
    statusText.textContent = message;
    status.className = `status ${type}`;
    status.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        status.classList.add('hidden');
    }, 5000);
}

// Reset Form
function resetForm() {
    selectedFile = null;
    imageInput.value = '';
    preview.src = '';
    preview.classList.add('hidden');
    uploadBox.querySelector('.upload-content').classList.remove('hidden');
    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
}






