async function runInference(img) {
    let canvas = document.createElement('canvas');
    document.body.append(canvas);
    canvas.width = img.width;
    canvas.height = img.height;

    const pre = document.createElement('pre');
    document.body.append(pre);
    
    const model = await tf.automl.loadObjectDetection('model.json');
    const options = {score: 0.5, iou: 0.5, topk: 20};
    const predictions = await model.detect(img, options);
    console.log(predictions);
    pre.textContent = JSON.stringify(predictions, null, 2);
  
    var context = canvas.getContext('2d');
    context.drawImage(img, 0, 0);

    context.beginPath();
    for (var i in predictions) {
        var box = predictions[i].box
        context.rect(box.left, box.top, box.width, box.height);
        context.font = '24pt Arial'
        var text = predictions[i].label + ": " + predictions[i].score.toFixed(3);
        context.fillText(text, box.left, box.top, 200)
    }
    context.lineWidth = 4;
    context.stroke();
  
}

function readImages(files) {
    imageTypes = ['image/jpeg', 'image/png'];

    for (let i = 0; i < files.length; i++) {
        let file = files[i];
        console.log(file)

        if (imageTypes.indexOf(file.type) == -1) {
            continue;
        }

        let fileReader = new FileReader();
        fileReader.onload = function() {
            img = new Image();
            img.src = fileReader.result;

            img.addEventListener('load', function(e) {
                runInference(e.target);
            });
        }

        fileReader.readAsDataURL(file);
    }
}
