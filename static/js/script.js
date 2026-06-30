document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadContent = document.querySelector('.upload-content');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const loadingOverlay = document.getElementById('loading');
    
    // Result elements
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const confBar = document.getElementById('conf-bar');
    const heatmapImg = document.getElementById('heatmap-img');

    let currentFile = null;

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('dragover');
    }

    function unhighlight(e) {
        dropArea.classList.remove('dragover');
    }

    dropArea.addEventListener('drop', handleDrop, false);
    dropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                previewFile(file);
                analyzeBtn.disabled = false;
                // Hide results if showing previous
                resultsSection.style.display = 'none';
            } else {
                alert("Please upload an image file.");
            }
        }
    }

    function previewFile(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            imagePreview.style.display = 'block';
            uploadContent.style.opacity = '0';
        }
    }

    // Handle Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('image', currentFile);

        // Show loading state
        loadingOverlay.style.display = 'flex';
        analyzeBtn.disabled = true;
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                // Populate results
                predClass.textContent = data.prediction;
                predConf.textContent = data.confidence;
                
                // Animate confidence bar
                const confValue = parseFloat(data.confidence);
                confBar.style.width = '0%';
                setTimeout(() => {
                    confBar.style.width = `${confValue}%`;
                }, 100);

                if (data.heatmap) {
                    heatmapImg.src = `data:image/jpeg;base64,${data.heatmap}`;
                } else {
                    heatmapImg.src = imagePreview.src; // fallback
                }

                resultsSection.style.display = 'block';
            }
        } catch (error) {
            console.error('Error during prediction:', error);
            alert("An error occurred while communicating with the server.");
        } finally {
            loadingOverlay.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });
});
