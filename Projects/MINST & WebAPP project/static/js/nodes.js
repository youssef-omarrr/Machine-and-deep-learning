/* ========== ELEMENT REFERENCES ========== */
// DOM references to the areas where nodes and connecting lines will be placed
const resultArea = document.getElementById("result-area");
const linesArea = document.getElementById("lines-area");

/* ========== CREATE NODES AND LINES ========== */
// Dynamically create 10 circular nodes and 10 lines connecting from canvas to each node
for (var i = 0; i < 10; i++) {
    // Create a wrapper for each node to allow for better layout control
    const wrapper = document.createElement("div");
    wrapper.className = "node-wrapper";
    wrapper.id = "node-wrapper" + i;

    var node = document.createElement("div");
    node.className = "node";
    node.id = "node" + i;

    // Number inside the circle
    var p = document.createElement("p");
    p.id = "p" + i;
    p.textContent = i;
    node.appendChild(p);

    // Probability outside (to the right) of the circle
    var prob = document.createElement("span");
    prob.className = "probability";
    prob.id = "prob" + i;
    prob.textContent = "";  // initially blank
    node.appendChild(prob);

    // Combine circle and probability in wrapper
    wrapper.appendChild(node);
    wrapper.appendChild(prob);

    resultArea.appendChild(wrapper);

    var line = document.createElement("div");
    line.className = "node-line";
    line.id = "line" + i;
    linesArea.appendChild(line);
}


/* ========== POSITION LINES ========== */
// Calculates angle and length between canvas and each node, and draws a connecting line
function positionLines() {
    const nodes = resultArea.querySelectorAll('.node');
    const canvas = document.getElementById('canvas');
    const canvasRect = canvas.getBoundingClientRect();
    const linesAreaRect = linesArea.getBoundingClientRect();

    const startX = canvasRect.right - linesAreaRect.left;
    const startY = canvasRect.top - linesAreaRect.top + canvasRect.height / 2;

    const nodeRadius = 25;

    nodes.forEach((node, i) => {
        const line = document.getElementById('line' + i);
        const nodeRect = node.getBoundingClientRect();

        const endX = nodeRect.left - linesAreaRect.left + nodeRect.width / 2;
        const endY = nodeRect.top - linesAreaRect.top + nodeRect.height / 2;

        const dx = endX - startX;
        const dy = endY - startY;
        const length = Math.sqrt(dx * dx + dy * dy);

        const shortenedLength = Math.max(0, length - nodeRadius);
        const angle = Math.atan2(dy, dx) * 180 / Math.PI;

        line.style.left = startX + 'px';
        line.style.top = (startY - 2) + 'px';
        line.style.width = shortenedLength + 'px';
        line.style.transform = `rotate(${angle}deg)`;
        line.style.transformOrigin = '0 50%';
    });
}


/* ========== EVENT LISTENERS FOR LAYOUT UPDATES ========== */
// Ensures lines are correctly positioned on load and window resize
window.addEventListener('DOMContentLoaded', function() {
    positionLines();
});

window.addEventListener('resize', function() {
    positionLines();
});
