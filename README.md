# Hot Dog / Not Hot Dog in TensorFlow.js

Tensorflow-js implementation of the Hot Dog / Not Hot Dog model, with an improvement.

![hotdog](images/20180405185342.jpg)

## How it works

The model is trained to distinguish hot dogs from... not hot dogs.

But instead of just showing the classifier label, we plot the last convolutional layer before global average pooling. In plain terms, we train the network to say "hot dog" and also show roughly where in the image the hot dog is.

For more information how it works and how to train your own model, see the [ディープラーニングによるホットドッグ検出器のレシピ](http://techlife.cookpad.com/entry/2018/04/06/124455) blog post (forgive me my poor Japanese).

## Run it

You'll need a browser, a webcam and some way to run a local web serve for serving static files (can use `npx http-server`). And a hot dog.

1. Clone or download the contents of this repository
2. Run a local static file web server
3. Check out `index.html` in your browser, give it　permission to your webcam
4. Show a hot dog to your camera and be amazed by the framerate of a deep learning model running in your browser

## Lazy version

Just see it online [HERE](https://tokyo-ml.github.io/hotdog-tf-js/). It's hosted on github, because it's all static files!

## Disclaimer

