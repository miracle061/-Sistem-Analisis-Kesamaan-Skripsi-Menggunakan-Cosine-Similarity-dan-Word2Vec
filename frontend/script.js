const backendURL = "http://127.0.0.1:5000";

async function uploadPDF() {
    const fileInput = document.getElementById("pdfInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a PDF file");
        return;
    }

    document.getElementById("loading").classList.remove("hidden");

    let formData = new FormData();
    formData.append("file", file);

    // 1. Upload the file
    const uploadRes = await fetch(`${backendURL}/upload`, {
        method: "POST",
        body: formData
    });

    const uploadData = await uploadRes.json();

    if (!uploadData.filename) {
        alert("Upload failed");
        return;
    }

    const filename = uploadData.filename;

    // 2. Get similarity results
    const simRes = await fetch(`${backendURL}/similar/${filename}`);
    const simData = await simRes.json();

    document.getElementById("loading").classList.add("hidden");

    // 3. Show results
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    simData.forEach(item => {
        resultsDiv.innerHTML += `
            <div class="result-item">
                <b>${item.file_name}</b><br>
                Similarity: ${(item.similarity * 100).toFixed(2)}%<br><br>
                <button onclick="previewPDF('${item.file_name}')">Preview PDF</button>
            </div>
        `;
    });
}

function previewPDF(filename) {
    window.open(`${backendURL}/preview/${filename}`, "_blank");
}
