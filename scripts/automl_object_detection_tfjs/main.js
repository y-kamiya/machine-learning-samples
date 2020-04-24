async function run() {
    const img = document.getElementById('image');
  
    const canvas = document.getElementById('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    
    const model = await tf.automl.loadObjectDetection('model.json');
    const options = {score: 0.5, iou: 0.5, topk: 20};
    const predictions = await model.detect(img, options);
    console.log(predictions);
    // Show the resulting object on the page.
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(predictions, null, 2);
    document.body.append(pre);
  
    var context = canvas.getContext('2d');
    context.beginPath();
    for (var i in predictions) {
        var box = predictions[i].box
        var label = 
        context.rect(box.left, box.top, box.width, box.height);
        context.font = '24pt Arial'
        var text = predictions[i].label + ": " + predictions[i].score.toFixed(3);
        context.fillText(text, box.left, box.top, 200)
    }
    context.lineWidth = 4;
    context.stroke();
  
}

