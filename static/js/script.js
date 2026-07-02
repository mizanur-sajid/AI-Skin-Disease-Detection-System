/**
 * DermAI Diagnostix — Frontend Logic
 * Handles upload, prediction, gauge animation, comparison tabs, toast system, and disease info.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── DOM References ────────────────────────────────────────
    const dropArea       = document.getElementById('drop-area');
    const fileInput      = document.getElementById('file-input');
    const imagePreview   = document.getElementById('image-preview');
    const uploadContent  = document.getElementById('upload-content');
    const previewOverlay = document.getElementById('preview-overlay');
    const previewFilename = document.getElementById('preview-filename');
    const changeImageBtn = document.getElementById('change-image-btn');
    const analyzeBtn     = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const loadingOverlay = document.getElementById('loading-overlay');
    const toastContainer = document.getElementById('toast-container');

    // Result elements
    const predClass          = document.getElementById('pred-class');
    const gaugeFill          = document.getElementById('gauge-fill');
    const gaugeValue         = document.getElementById('gauge-value');
    const confidenceText     = document.getElementById('confidence-text');
    const riskBadgeContainer = document.getElementById('risk-badge-container');
    const diseaseDescription = document.getElementById('disease-description');
    const heatmapImg         = document.getElementById('heatmap-img');
    const originalCompareImg = document.getElementById('original-compare-img');

    // Comparison tabs
    const comparisonTabs = document.getElementById('comparison-tabs');

    let currentFile = null;

    // ── Disease Info Database ─────────────────────────────────
    const DISEASE_INFO = {
        'Actinic keratoses (akiec)': {
            description: 'Actinic keratoses are rough, scaly patches caused by years of sun exposure. They are considered precancerous and can potentially progress to squamous cell carcinoma if left untreated.',
            risk: 'moderate'
        },
        'Basal cell carcinoma (bcc)': {
            description: 'Basal cell carcinoma is the most common form of skin cancer. It grows slowly and rarely spreads, but can cause significant local tissue damage if not treated early.',
            risk: 'high'
        },
        'Benign keratosis-like lesions (bkl)': {
            description: 'Benign keratosis-like lesions include seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses. These are non-cancerous growths that typically don\'t require treatment.',
            risk: 'low'
        },
        'Dermatofibroma (df)': {
            description: 'Dermatofibromas are common benign skin nodules, often found on the legs. They are firm, small, and usually harmless. They may result from minor injuries like insect bites.',
            risk: 'low'
        },
        'Melanoma (mel)': {
            description: 'Melanoma is the most dangerous form of skin cancer. It develops from melanocytes and can spread rapidly to other organs. Early detection is critical for successful treatment.',
            risk: 'high'
        },
        'Melanocytic nevi (nv)': {
            description: 'Melanocytic nevi (moles) are benign growths of melanocytes. They are very common and usually harmless, though atypical moles may require monitoring for changes.',
            risk: 'low'
        },
        'Vascular lesions (vasc)': {
            description: 'Vascular lesions include cherry angiomas, angiokeratomas, and pyogenic granulomas. Most are benign conditions involving blood vessels in the skin.',
            risk: 'low'
        }
    };

    // ── Toast Notification System ─────────────────────────────
    function showToast(message, type = 'info') {
        const iconMap = {
            error:   'fa-solid fa-circle-exclamation',
            success: 'fa-solid fa-circle-check',
            info:    'fa-solid fa-circle-info'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <i class="toast-icon ${iconMap[type] || iconMap.info}"></i>
            <span class="toast-message">${message}</span>
            <button class="toast-close" aria-label="Close notification"><i class="fa-solid fa-xmark"></i></button>
        `;

        toastContainer.appendChild(toast);

        // Close button
        toast.querySelector('.toast-close').addEventListener('click', () => removeToast(toast));

        // Auto-dismiss after 5 seconds
        setTimeout(() => removeToast(toast), 5000);
    }

    function removeToast(toast) {
        if (!toast.parentElement) return;
        toast.classList.add('removing');
        toast.addEventListener('animationend', () => toast.remove());
    }

    // ── Drag & Drop ───────────────────────────────────────────
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        handleFiles(e.dataTransfer.files);
    }, false);

    dropArea.addEventListener('click', (e) => {
        // Don't trigger file dialog if clicking the change button
        if (e.target.closest('#change-image-btn')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    // Change image button
    changeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
        fileInput.click();
    });

    // ── File Handling ─────────────────────────────────────────
    function handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        if (!file.type.startsWith('image/')) {
            showToast('Please upload a valid image file (JPG, PNG, WEBP).', 'error');
            return;
        }

        currentFile = file;
        previewFile(file);
        analyzeBtn.disabled = false;

        // Hide previous results
        resultsSection.classList.add('hidden');
    }

    function previewFile(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            imagePreview.src = reader.result;
            imagePreview.classList.add('visible');
            uploadContent.style.display = 'none';

            // Show filename overlay
            previewFilename.querySelector('span').textContent = truncateFilename(file.name, 25);
        };
    }

    function resetUpload() {
        currentFile = null;
        imagePreview.src = '#';
        imagePreview.classList.remove('visible');
        uploadContent.style.display = '';
        analyzeBtn.disabled = true;
        fileInput.value = '';
    }

    function truncateFilename(name, maxLen) {
        if (name.length <= maxLen) return name;
        const ext = name.split('.').pop();
        return name.substring(0, maxLen - ext.length - 4) + '….' + ext;
    }

    // ── Confidence Gauge Animation ────────────────────────────
    const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 45; // ~283

    function animateGauge(percent) {
        const offset = GAUGE_CIRCUMFERENCE - (percent / 100) * GAUGE_CIRCUMFERENCE;

        // Reset first for re-animation
        gaugeFill.style.transition = 'none';
        gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                gaugeFill.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)';
                gaugeFill.style.strokeDashoffset = offset;
            });
        });

        // Animate the counter
        animateCounter(gaugeValue, 0, percent, 1200);
    }

    function animateCounter(element, start, end, duration) {
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * eased);
            element.textContent = `${current}%`;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // ── Comparison Tabs ───────────────────────────────────────
    comparisonTabs.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (!btn) return;

        // Update active tab
        comparisonTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const tab = btn.dataset.tab;
        if (tab === 'heatmap') {
            heatmapImg.classList.remove('hidden');
            originalCompareImg.classList.add('hidden');
        } else {
            heatmapImg.classList.add('hidden');
            originalCompareImg.classList.remove('hidden');
        }
    });

    // ── Risk Badge ────────────────────────────────────────────
    function renderRiskBadge(riskLevel) {
        const labels = {
            low: { icon: 'fa-shield-halved', text: 'Low Risk' },
            moderate: { icon: 'fa-triangle-exclamation', text: 'Moderate Risk' },
            high: { icon: 'fa-circle-exclamation', text: 'High Risk' }
        };

        const info = labels[riskLevel] || labels.low;
        riskBadgeContainer.innerHTML = `
            <span class="risk-badge ${riskLevel}">
                <i class="fa-solid ${info.icon}"></i>
                ${info.text}
            </span>
        `;
    }

    // ── Confidence Description ────────────────────────────────
    function getConfidenceDescription(percent) {
        if (percent >= 90) return 'Very high confidence — the model is strongly certain about this classification.';
        if (percent >= 70) return 'High confidence — the model is fairly certain about this prediction.';
        if (percent >= 50) return 'Moderate confidence — consider consulting a specialist for verification.';
        return 'Low confidence — the model is uncertain. A professional evaluation is recommended.';
    }

    // ── Analyze Button ────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('image', currentFile);

        // Show loading
        loadingOverlay.classList.add('active');
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            // ── Populate Results ──
            // Prediction name
            predClass.textContent = data.prediction;

            // Confidence gauge
            const confPercent = parseFloat(data.confidence);
            animateGauge(confPercent);

            // Confidence description
            confidenceText.textContent = getConfidenceDescription(confPercent);

            // Disease info
            const diseaseInfo = DISEASE_INFO[data.prediction];
            if (diseaseInfo) {
                diseaseDescription.textContent = diseaseInfo.description;
                renderRiskBadge(diseaseInfo.risk);
            } else {
                diseaseDescription.textContent = 'Detailed information for this condition is not available in the current database.';
                riskBadgeContainer.innerHTML = '';
            }

            // Grad-CAM heatmap
            if (data.heatmap) {
                heatmapImg.src = `data:image/jpeg;base64,${data.heatmap}`;
            } else {
                heatmapImg.src = imagePreview.src; // fallback
            }

            // Original image for comparison
            originalCompareImg.src = imagePreview.src;

            // Reset to heatmap tab
            comparisonTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            comparisonTabs.querySelector('[data-tab="heatmap"]').classList.add('active');
            heatmapImg.classList.remove('hidden');
            originalCompareImg.classList.add('hidden');

            // Show results
            resultsSection.classList.remove('hidden');

            // Smooth scroll to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 200);

            showToast('Analysis complete! Review the diagnosis report below.', 'success');

        } catch (error) {
            console.error('Prediction error:', error);
            showToast('Failed to communicate with the server. Please try again.', 'error');
        } finally {
            loadingOverlay.classList.remove('active');
            analyzeBtn.disabled = false;
        }
    });
});
