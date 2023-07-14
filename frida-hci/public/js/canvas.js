function loadCanvas(path, callback) {
    // Read canvas
    const image = new Image();
    image.src = path + '?t=' + Date.now();

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    image.onload = function() {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        callback();
    };
}

function canvasreset() {
    loadCanvas('data/stable/start.jpg?t=' + Date.now(), function (){});
    document.getElementById('canvas').style.setProperty('opacity', 1);
}

function canvasinit() {
    started = false;
    const canvas = document.getElementById('canvas');

    // Function called when mouse enters the canvas
    function handleMouseEnter() {
        if (!started) {
            document.body.classList.add("drawing");
        }
      // Perform actions when entering the canvas
    }
    
    // Function called when mouse leaves the canvas
    function handleMouseLeave() {
        document.body.classList.remove("drawing");
        isPainting = false;
      // Perform actions when leaving the canvas
    }
    
    // Attach event listeners
    canvas.addEventListener('mouseenter', handleMouseEnter);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    /* PAINTING */
    const context = canvas.getContext('2d', { willReadFrequently: true });

    const minPixelSize = 4;
    const inkColor = 'black';

    canvas.width = 512;
    canvas.height = 512;
    scale = canvas.width / canvas.offsetWidth;
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    // loadCanvas('data/init.jpg?t=' + Date.now());

    let isPainting = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    function startPainting(event) {
        isPainting = true;
        lastMouseX = (event.clientX - canvas.getBoundingClientRect().left) * scale;
        lastMouseY = (event.clientY - canvas.getBoundingClientRect().top) * scale;
    }

    function stopPainting() {
        isPainting = false;

        // if (typeof global_counter !== 'undefined') {
        //     global_counter += 1;
        //     document.getElementById('global-counter').innerHTML = global_counter;
        // }
    }

    function paintPixel(event) {
        if (!isPainting || started) return;

        const rect = canvas.getBoundingClientRect();
        const mouseX = (event.clientX - rect.left) * scale;
        const mouseY = (event.clientY - rect.top) * scale;

        // Calculate the distance between the current mouse position and the previous position
        const distance = Math.sqrt(
            Math.pow(mouseX - lastMouseX, 2) + Math.pow(mouseY - lastMouseY, 2)
        );

        // Interpolate between the last position and the current position
        const steps = Math.floor(distance / minPixelSize + 1) * 2;
        const deltaX = (mouseX - lastMouseX) / steps;
        const deltaY = (mouseY - lastMouseY) / steps;

        pixelSize = Math.max(minPixelSize, Math.pow(deltaX, 2) + Math.pow(deltaY, 2) * 3)
        for (let i = 0; i < steps; i++) {
            const x = Math.floor(lastMouseX + i * deltaX);
            const y = Math.floor(lastMouseY + i * deltaY);

            const imageData = context.getImageData(x, y, pixelSize, pixelSize);
            const data = imageData.data;
            // Iterate through each pixel
            for (let i = 0; i < data.length; i += 4) {
                // Extract the RGB values of the canvas pixel
                const canvasRed = data[i];
                const canvasGreen = data[i + 1];
                const canvasBlue = data[i + 2];
                // Extract the RGB values of the fill color
                const fillRed = 255;
                const fillGreen = 0;
                const fillBlue = 0;
                // Apply the linear burn blending formula
                const newRed = Math.max(canvasRed + fillRed - 255, 0);
                const newGreen = Math.max(canvasGreen + fillGreen - 255, 0);
                const newBlue = Math.max(canvasBlue + fillBlue - 255, 0);
                // Set the new pixel values
                data[i] = newRed;
                data[i + 1] = newGreen;
                data[i + 2] = newBlue;
            }
            // Put the modified image data back onto the canvas
            context.putImageData(imageData, x, y);
        }

        // Update the last recorded mouse position
        lastMouseX = mouseX;
        lastMouseY = mouseY;
    }

    canvas.addEventListener('mousedown', startPainting);
    canvas.addEventListener('mouseup', stopPainting);
    canvas.addEventListener('mousemove', paintPixel);
}

window.addEventListener('load', function() {
    canvasinit();
    canvasreset();
})