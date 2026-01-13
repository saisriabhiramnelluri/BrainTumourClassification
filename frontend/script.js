const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const predictBtn = document.getElementById('predict-btn');
const resultCard = document.getElementById('result-card');
const loader = document.getElementById('loader');
const resultContent = document.getElementById('result-content');
const iconContainer = document.querySelector('.icon-container');
const uploadText = document.querySelector('.upload-text');

// REPLACE THIS WITH YOUR RENDER BACKEND URL AFTER DEPLOYMENT
const API_URL = 'https://braintumourclassification.onrender.com/predict';

// Drag & Drop Interactions
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.add('visible');
            iconContainer.classList.add('hidden');
            uploadText.classList.add('hidden');
            predictBtn.disabled = false;

            // Store file for upload
            dropZone.file = file;
        };
        reader.readAsDataURL(file);

        // Hide previous results
        resultCard.classList.add('hidden');
    }
}

predictBtn.addEventListener('click', async () => {
    if (!dropZone.file) return;

    // UI Updates
    predictBtn.disabled = true;
    dropZone.classList.add('scanning'); // Start scanning animation
    resultCard.classList.remove('hidden');
    loader.classList.remove('hidden');
    resultContent.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', dropZone.file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Update Results
        document.getElementById('prediction-text').textContent = data.class.toUpperCase();
        document.getElementById('confidence-text').textContent = data.confidence;
        document.getElementById('prediction-text').style.color =
            data.class === 'notumor' ? '#4ade80' : '#f87171'; // Green for safe, Red for tumor

    } catch (error) {
        alert("Error analyzing image. Make sure the backend is running!");
        console.error(error);
    } finally {
        // Stop animations
        dropZone.classList.remove('scanning');
        loader.classList.add('hidden');
        resultContent.classList.remove('hidden');
        predictBtn.disabled = false;
    }
});
