const form = document.getElementById("uploadForm");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const spinner = document.getElementById("spinner");

document.getElementById("imageInput").addEventListener("change", function(e) {
    const reader = new FileReader();
    reader.onload = function() {
        preview.src = reader.result;
        preview.style.display = "block";
    }
    reader.readAsDataURL(e.target.files[0]);
});

form.addEventListener("submit", async function(e) {
    e.preventDefault();

    spinner.style.display = "block";
    resultDiv.innerHTML = "";

    const formData = new FormData();
    formData.append("image", document.getElementById("imageInput").files[0]);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    spinner.style.display = "none";

    resultDiv.innerHTML = `
        <h3>${data.prediction}</h3>
        <p>Confidence: ${data.confidence}%</p>
        <div class="progress">
            <div class="progress-bar bg-success" 
                 style="width: ${data.confidence}%">
            </div>
        </div>
    `;
});