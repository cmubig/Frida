global_counter = 0

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function recurse_render(n_strokes, counter) {
    if (!started) return;
    if (counter >= n_strokes) {
        start();
        return;
    }
    painting = true;
    is_first = false;

    document.body.classList.remove('state-thinking');
    document.body.classList.add('state-painting');
    pbar = document.getElementById('progress-bar');
    document.getElementById('main-button').disabled = false;
    for (let i = 1; i <= 20; i++) {
        if (!started) return;
        pbar.setAttribute('style', '--percent: ' + i * 5 + ';');
        await sleep(25);
    }

    fetch('/render?counter=' + counter)
        .then(data => {
            if (started) {
                loadCanvas('data/camera.jpg?t=' + counter, function() {
                    global_counter += 1;
                    document.getElementById('global-counter').innerHTML = global_counter;
                    if (global_counter >= 196) {
                        document.getElementById('main-button').disabled = true;
                        return;
                    }
                    recurse_render(n_strokes, counter + 1);
                });
            }
        });
}

function manifest() {
    img = document.getElementById('thumbnail');
    img.style.setProperty('z-index', '0');
    img.style.setProperty('visibility', 'hidden');

    canvas = document.getElementById('canvas');
    var dataURL = canvas.toDataURL('image/jpeg', 1.0);
    var formData = new FormData();
    formData.append('image_data', dataURL);
    $.ajax({
        type: "POST",
        url: "/save_camera",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            prompt = document.getElementById('prompt').value;
            fetch('/manifest?prompt=' + prompt +
                            '&iter=1')
                .then(data => {
                    img = document.getElementById('thumbnail');
                    img.setAttribute('src', 'data/stable/1.jpg?t=' + new Date().getUTCMilliseconds());
                    img.style.setProperty('z-index', '10');
                    img.style.setProperty('visibility', 'visible');

                    if (tasks[task_counter] == 'skip') {
                        img.setAttribute('onclick',"select();");
                        document.getElementById('image-container').classList.remove('disabled');
                    } else if (tasks[task_counter] == 'user') {
                        select();
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Handle any errors that occur during the request
                });
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

function select() {
    if (tasks[task_counter] == 'user') {
        document.getElementById('image-container').classList.add('disabled');

        fetch('/select-objective?objective-id=1')
        .then(data => {
            start();
        })
        .catch(error => {
            console.error('Error:', error);
            // Handle any errors that occur during the request
        });
    } else if (tasks[task_counter] == 'skip') {
        loadCanvas('data/stable/1.jpg', function(){});
    }
}

function start() {
    // Update button
    button = document.getElementById('main-button');
    button.innerHTML = "<i class=\"fa-solid fa-pause\"></i> pause FRIDA";
    document.body.classList.add("started");
    started = true;
    button.setAttribute('onclick', 'stop()');
    document.getElementById('reset-canvas').disabled = true;

    /* Save canvas to file */
    canvas = document.getElementById('canvas');
    var dataURL = canvas.toDataURL('image/jpeg', 1);
    var formData = new FormData();
    formData.append('image_data', dataURL);
    $.ajax({
        type: "POST",
        url: "/save_camera",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            painting = false;
            prompt = '';
            // prompt = document.getElementById('prompt').value;
            fetch('/think?poll=1' +
                         '&task=' + tasks[task_counter] +
                         '&counter=' + global_counter)
                .then(response => response.json())
                .then(response => {
                    if (response['plan_exists'])
                        startPBar(30);
                    else
                        startPBar(500);


                    fetch('/think?counter=' + global_counter + 
                          '&is-first=' + is_first +
                          '&task=' + tasks[task_counter] +
                          '&prompt=' + prompt +
                          '&poll=0')
                        .then(response => response.json())
                        .then(data => {
                            recurse_render(data.n_strokes, 0);
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                })
                .catch(error => {
                    console.error('Error:', error);
                });

            async function startPBar(duration) {
                document.body.classList.remove('state-painting');
                document.body.classList.add('state-thinking');
                pbar = document.getElementById('progress-bar');
                document.getElementById('main-button').disabled = true;
                tbins = duration * 1;
                for (let i = 1; i <= tbins; i++) {
                    if (!started || painting) return;

                    pbar.setAttribute('style', '--percent: ' + i * (100.0/tbins) + ';');
                    await sleep(duration * 1000.0 / tbins);
                }
            }
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

function stop() {
    // Reset button
    button = document.getElementById('main-button');
    button.innerHTML = "<i class=\"fa-solid fa-play\"></i> resume FRIDA";
    document.body.classList.remove("started");
    started = false;
    button.setAttribute('onclick', 'start()');
    document.getElementById('reset-canvas').disabled = false;

    // Remove pbar
    document.body.classList.remove('state-thinking');
    document.body.classList.remove('state-painting');
    document.getElementById('progress-bar').setAttribute('style', '--percent: 0;');

    loadCanvas('data/camera.jpg?t=' + Date.now());
}

function export_canvas() {
    canvas = document.getElementById('canvas');
    var dataURL = canvas.toDataURL('image/jpeg', 1.0);
    var formData = new FormData();
    formData.append('image_data', dataURL);
    $.ajax({
        type: "POST",
        url: "/save_camera",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            // Create a temporary anchor element
            var tempAnchor = document.createElement('a');
            tempAnchor.href = 'data/camera.jpg';
            tempAnchor.setAttribute('download', '');

            // Programmatically trigger the click event
            var clickEvent = new MouseEvent('click', {
                view: window,
                bubbles: true,
                cancelable: true
            });
            tempAnchor.dispatchEvent(clickEvent);
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

function reset_counter() {
    global_counter = 0;
    document.getElementById('global-counter').innerHTML = global_counter;
    is_first = true;
}

function taskinit() {
    task_counter = 0;
    tasks = ['skip', 'user'];
    document.getElementById('frida-header').setAttribute('task', tasks[task_counter]);

    if (tasks[task_counter] != 'skip' && tasks[task_counter] != 'user') {
        document.getElementById('thumbnail').setAttribute('src', 'data/objectives/' + tasks[task_counter] + '.jpg?t=' + new Date().getUTCMilliseconds());
    }
}
function swap_task() {
    task_counter += 1;
    if (task_counter == tasks.length) {
        task_counter = 0;
    }

    document.getElementById('frida-header').setAttribute('task', tasks[task_counter]);
    if (tasks[task_counter] != 'skip' && tasks[task_counter] != 'user') {
        document.getElementById('thumbnail').setAttribute('src', 'data/objectives/' + tasks[task_counter] + '.jpg?t=' + new Date().getUTCMilliseconds());
    }
    
    counter = 0;
    global_counter = 0;
    document.getElementById('global-counter').innerHTML = global_counter;
}

window.addEventListener('load', function() {
  taskinit();
  is_first = true;
})