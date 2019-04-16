// first deal with browser prefixes
var getUserMedia = navigator.getUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.webkitGetUserMedia;

// make sure it's supported and bind to navigator
if (getUserMedia) {
    getUserMedia = getUserMedia.bind(navigator);
} else {
    // have to figure out how to handle the error somehow
}

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
    grayscale = tf.onesLike(heatmap).sub(heatmap).mul(grayscale).squeeze().mul(tf.scalar(0.3))
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


var app = new Vue({
  el: '#app',
  data: {
    model: null,
    video: null,
    webcam: null,
    offscreen: null,
    onscreen: null,
    url: '',
    playing: false,
    loopFrame: null
  },

  computed: {
    message: function() {
        if (!this.model) {
            return "Loading the model"
        } else {
            if (!this.playing) {
                return "Tap to start"
            } else {
                return "Go find a hot dog!"
            }
        }
    },
    buttonText: function() {
        return this.playing ? "Pause" : "Play"
    }
  },

  methods: {
    loadModel: function() {
        return tf.loadLayersModel('model/model.json').then(loadedModel => {
            this.model = loadedModel
            return loadedModel
        })
    },

    getCamera: function() {
        var webcam = this.video
        return new Promise(function(resolve, reject) {
          getUserMedia(
            {
              video: {
                height: 240,
                width: 320,
                facingMode: 'user',
              }, audio: false},
            stream => {
              webcam.srcObject = stream;
              this.url = stream
              webcam.onloadedmetadata = (e) => {
                webcam.width = webcam.videoWidth;
                webcam.height = webcam.videoHeight;
                return webcam.play().then(resolve)
              }
            },
            error => reject
          )
        })
    },


    togglePlay: function() {
        // exit if clicked too early
        if (!this.video) {
            return
        }
        if (this.playing) {
            this.video.pause()
            this.playing = !this.playing
        } else {
            this.video.play()
            this.playing = !this.playing
        }
    },

    continue: function() {
        this.loopFrame = requestAnimationFrame(this.loop)
    },

    loop: async function() {
        if (!this.model || !this.playing) {
            this.continue()
            return
        }
        // copy camera image to convas
        this.offscreen.drawImage(this.video, 0, 0, 640, 480)

        // read the pixels from canvas
        var imageData = this.offscreen.getImageData(0, 0, 640, 480)
        var pixeldata = tf.browser.fromPixels(imageData)

        // use the model to predict response
        var response = await tf.tidy(() => this.model.predict(preprocess(pixeldata)))
        responseData = await postprocess(response, pixeldata, 640, 480).data()

        // copy the loaded tensor to imageData
        for (var i = 0; i < responseData.length; i+=1) {
            imageData.data[i] = responseData[i]
        }

        // paste the result on screen
        this.onscreen.putImageData(imageData, 0, 0)
        this.continue()
    }

  },

  created: function() {
    this.video = document.getElementById("the_video")

    this.loadModel()
      .then(this.getCamera)
      .then(() => {
        // get the offscreen canvas
        var offscreenCanvas = document.createElement("canvas")
        offscreenCanvas.width = 640
        offscreenCanvas.height = 480
        this.offscreen = offscreenCanvas.getContext("2d")

        // get the onscreen canvas
        var onscreenCanvas = document.getElementById("the_canvas")
        this.onscreen = onscreenCanvas.getContext("2d")

        // start the loop
        this.continue()

    })
  }

})
