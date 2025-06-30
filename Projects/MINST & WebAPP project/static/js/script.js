const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');
let drawing = false;


canvas.addEventListener('mousedown', ()=> drawing= true);
canvas.addEventListener('mouseup', ()=> drawing= false);
canvas.addEventListener('mouseleave', () => drawing = false);  // optional: stop drawing if mouse leaves
canvas.addEventListener('mousemove', draw);


function draw(event){
    if (!drawing) 
        return;

    ctx.beginPath();
    ctx.arc(event.offsetX, event.offsetY, 8, 0, 2 * Math.PI);  // radius 8–10 is ideal
    ctx.fillStyle = 'black';
    ctx.fill();

}


function clearCanvas(){
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawing = false;  // reset state
}


function predict() {
    const resultEl = document.getElementById('result');
    const previewImg = document.getElementById('preview-img');

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'image1.png');

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.Predicted !== undefined) {
                resultEl.textContent = "Prediction: " + data.Predicted;

                // Show the saved image with cache-busting
                previewImg.src = '/static/received_input.png?t=' + new Date().getTime();
                previewImg.style.display = 'block';
            } else {
                resultEl.textContent = "Error: Invalid response from server.";
            }
        })
        .catch(error => {
            resultEl.textContent = "Error: " + error;
        });
    }, 'image/png');
}


window.onload = clearCanvas;