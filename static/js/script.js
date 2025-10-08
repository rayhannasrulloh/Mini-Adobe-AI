document.addEventListener('DOMContentLoaded', function () {
    const imageUpload = document.getElementById('imageUpload');
    const imageCanvas = document.getElementById('imageCanvas');
    const spinner = document.getElementById('spinner');
    const undoBtn = document.getElementById('undoBtn');
    const redoBtn = document.getElementById('redoBtn');
    const resetBtn = document.getElementById('resetBtn');

    let originalFilename = null;
    let currentFilename = null;
    let history = [];
    let historyIndex = -1;

    // --- State Management ---
    function showSpinner() {
        spinner.style.display = 'block';
    }

    function hideSpinner() {
        spinner.style.display = 'none';
    }

    function updateButtons() {
        undoBtn.disabled = historyIndex <= 0;
        redoBtn.disabled = historyIndex >= history.length - 1;
        resetBtn.disabled = !originalFilename;
    }

    function pushToHistory(filename) {
        // If we are in the middle of history, clear the future
        if (historyIndex < history.length - 1) {
            history = history.slice(0, historyIndex + 1);
        }
        history.push(filename);
        historyIndex++;
        updateButtons();
    }

    // --- Event Listeners ---
    imageUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showSpinner();
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.filename) {
                originalFilename = data.filename;
                currentFilename = data.filename;
                imageCanvas.src = `/uploads/${currentFilename}`;
                history = [currentFilename];
                historyIndex = 0;
                updateButtons();
            } else {
                alert('Error uploading file: ' + data.error);
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('An error occurred during upload.');
        } finally {
            hideSpinner();
        }
    });

    // Event listener for sliders (on change, not on input, to avoid too many requests)
    document.querySelectorAll('.tool-slider').forEach(slider => {
        slider.addEventListener('change', (e) => {
            const operation = e.target.dataset.op;
            const value = e.target.value;
            applyOperation(operation, value);
        });
    });

    // Event listener for buttons
    document.querySelectorAll('.tool-btn').forEach(button => {
        button.addEventListener('click', (e) => {
            const operation = e.target.dataset.op;
            const value = e.target.dataset.value;
            applyOperation(operation, value);
        });
    });
    
    // Undo/Redo/Reset Logic
    undoBtn.addEventListener('click', () => {
        if (historyIndex > 0) {
            historyIndex--;
            currentFilename = history[historyIndex];
            imageCanvas.src = `/uploads/${currentFilename}`;
            updateButtons();
        }
    });
    
    redoBtn.addEventListener('click', () => {
        if (historyIndex < history.length - 1) {
            historyIndex++;
            currentFilename = history[historyIndex];
            imageCanvas.src = `/uploads/${currentFilename}`;
            updateButtons();
        }
    });
    
    resetBtn.addEventListener('click', () => {
        if (originalFilename) {
            currentFilename = originalFilename;
            imageCanvas.src = `/uploads/${currentFilename}`;
            history = [originalFilename];
            historyIndex = 0;
            updateButtons();
        }
    });

    // --- Core API Call Function ---
    async function applyOperation(operation, value) {
        if (!currentFilename) {
            alert('Please upload an image first.');
            return;
        }

        showSpinner();
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: currentFilename,
                    operation: operation,
                    value: value
                })
            });

            const data = await response.json();
            if (data.filename) {
                currentFilename = data.filename;
                imageCanvas.src = `/uploads/${currentFilename}?t=${new Date().getTime()}`; // Cache buster
                pushToHistory(currentFilename);
            } else {
                alert('Error processing image: ' + data.error);
            }
        } catch (error) {
            console.error('Processing error:', error);
            alert('An error occurred during processing.');
        } finally {
            hideSpinner();
        }
    }
});