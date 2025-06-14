document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const scamForm = document.getElementById('scamForm');
    const queryInput = document.getElementById('queryInput');
    const resultDiv = document.getElementById('resultContainer');

    // Drag & Drop functionality
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateDropZoneLabel();
            }
        });
        fileInput.addEventListener('change', updateDropZoneLabel);
    }

    function updateDropZoneLabel() {
        if (fileInput.files[0]) {
            dropZone.querySelector('.drop-message span').textContent = fileInput.files[0].name;
        } else {
            dropZone.querySelector('.drop-message span').textContent = "Click to upload or drag and drop";
        }
    }

    // Form submission - Always use FormData for single endpoint
    scamForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultDiv.innerHTML = '';
        const text = queryInput.value.trim();
        const file = fileInput.files[0];

        if (!text && !file) {
            showError("Please enter a message/link or upload a file.");
            return;
        }

        try {
            showLoading();
            const formData = new FormData();

            if (file) {
                formData.append('file', file);
            }
            if (text && !file) {
                formData.append('text', text);
            }

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            displayResult(result, file);
        } catch (error) {
            console.error("Error:", error);
            showError("Analysis failed. Please try again.");
        }
    });

    function displayResult(data, file = null) {
        if (!resultDiv) return;

        if (data.error || (data.explanation && data.explanation.startsWith("Error:"))) {
            showError(data.explanation || data.error || "Unknown error occurred");
            return;
        }

        const label = data.label || (data.is_scam ? 'SCAM' : 'LEGITIMATE');
        const cardClass = label.toLowerCase() === 'scam' ? 'scam' : 'legit';
        const emoji = label === 'SCAM' ? '⚠️' : '✅';

        let imagePreviewHTML = '';
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                // Insert image preview above the result card
                resultDiv.innerHTML = `
                    <img src="${e.target.result}" alt="Uploaded Image" class="preview-image" style="max-width: 180px; margin-bottom: 16px; border-radius: 10px; box-shadow: 0 2px 12px rgba(127,133,245,0.18);">
                    ${resultDiv.innerHTML}
                `;
            };
            reader.readAsDataURL(file);
        }

        resultDiv.innerHTML = `
            <div class="result-card ${cardClass}">
                <h3>${emoji} ${label}</h3>
                <p><strong>Type:</strong> ${formatScamType(data.scam_type || 'Unknown')}</p>
                <p><strong>Risk Score:</strong> ${data.risk_score !== undefined ? data.risk_score + '%' : 'N/A'}</p>
                ${data.explanation ? `<p><strong>Details:</strong> ${data.explanation}</p>` : ''}
                ${data.model_verified ? '<p><small>✅ Verified by AI model</small></p>' : ''}
            </div>
        `;
    }

    function formatScamType(scamType) {
        return scamType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    function showError(message) {
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error">${message}</div>`;
        }
    }

    function showLoading() {
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="loading">Analyzing...</div>`;
        }
    }
});
