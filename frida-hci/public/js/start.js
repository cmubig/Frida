function recurse_manifest(iter) {
    if (iter > 4) return;

    prompt = document.getElementById('prompt').value;

    fetch('/manifest?prompt=' + prompt +
                    '&iter=' + iter)
                .then(data => {
                    img = document.getElementById('img-v' + iter);
                    img.setAttribute('src', 'data/stable/' + iter + '.jpg?t=' + new Date().getUTCMilliseconds());
                    img.style.setProperty('z-index', '10');
                    img.style.setProperty('visibility', 'visible');

                    recurse_manifest(iter+1);
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Handle any errors that occur during the request
                });
}

function select(object_id) {
    src = document.getElementById('img-v' + object_id).getAttribute('src');
    if (!src) return;

    fetch('/select-objective?objective-id=' + object_id)
        .then(data => {
            window.location.href = '/';
        })
        .catch(error => {
            console.error('Error:', error);
            // Handle any errors that occur during the request
        });
}

function start(e) {
    e.preventDefault();
    document.getElementById('prompt').style.setProperty('margin-top', '20px');
    document.getElementById('image-section').style.setProperty('opacity', '1');

    for (let i = 1; i <= 4; i++) {
        img = document.getElementById('img-v' + i);
        img.removeAttribute('src', '');
        img.style.setProperty('visibility', 'hidden');
        img.style.setProperty('z-index', '0');
    }

    recurse_manifest(1);

    return false;
}