
function preprocess(pixeldata) {

    // resize image for the model
    var contents = tf.image.resizeBilinear(pixeldata, [224, 224], false)

    // convert to float and make a one-image batch
    contents = contents.toFloat().div(tf.scalar(255)).expandDims(0)

    return contents
}

function postprocess(tensor, pixeldata, width, height) {
    // heatmaps contain both "hotdog" and "not hot dog", pick "hotdog"
    var hotdog = 1
    var heatmap = tf.split(tensor.squeeze(), 2, 2)[hotdog]
    pixeldata = pixeldata.toFloat()

    // resize the heatmap to the same shape as the image
    heatmap = tf.image.resizeBilinear(heatmap, [height, width], false)

    // generate grayscale image
    var grayscale = pixeldata.mean(2).expandDims(2)

    // grayscale = grayscale * (1 - heatmap) * 0.5 (to darken)
    grayscale = tf.onesLike(heatmap).sub(heatmap).mul(grayscale).squeeze().mul(tf.scalar(0.5))
    // stack grayscale data on all 3 channels
    grayscaleStacked = tf.stack([grayscale, grayscale, grayscale]).transpose([1,2,0])

    // composite = image * heatmap  +  grayscale * (1-heatmap)
    composite = pixeldata.mul(heatmap).add(grayscaleStacked)

    // split RGB to add alpha
    var rgb = tf.split(composite, 3, 2)
    // alpha is all ones
    var alpha = tf.onesLike(rgb[0]).mul(tf.scalar(255))
    rgb.push(alpha)

    // join all channels
    var composite = tf.stack(rgb, 2)

    // convert to integer and return
    return composite.toInt()
}

(async function() {

    // the model will load later
    var model = null

    // create an offscreen canvas to draw on
    var offscreenCanvas = document.createElement("canvas")
    offscreenCanvas.width = 640
    offscreenCanvas.height = 480
    var offscreen = offscreenCanvas.getContext("2d")

    // get the onscreen canvas
    var onscreenCanvas = document.getElementById("the_canvas")
    var onscreen = onscreenCanvas.getContext("2d")

    loopFrame = null

    // get the video object
    var video = document.getElementById("the_video")

    // this loop will run every frame
    var loop = async function() {
        if (!model) {
            model = await tf.loadModel('model/model.json')
        }

        // copy camera image to convas
        offscreen.drawImage(video, 0, 0, 640, 480)

        // read the pixels from canvas
        var imageData = offscreen.getImageData(0, 0, 640, 480)
        var pixeldata = tf.fromPixels(imageData)

        // use the model to predict response
        var response = await tf.tidy(() => model.predict(preprocess(pixeldata)))
        responseData = await postprocess(response, pixeldata, 640, 480).data()

        // copy the loaded tensor to imageData
        for (var i = 0; i < responseData.length; i+=1) {
            imageData.data[i] = responseData[i]
        }

        // paste the result on screen
        onscreen.putImageData(imageData, 0, 0);
        loopFrame = requestAnimationFrame(loop);
    }

    // ask for camera access
    navigator.mediaDevices.getUserMedia({ audio: false, video: true })
        .then(stream => {
            video.src = window.URL.createObjectURL(stream)
            loopFrame = requestAnimationFrame(loop)
        })

})()


