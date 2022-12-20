/* MIT License

Copyright (c) 2022 P. Huttunen, H. Karstu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

var img = new Image();

var openFile = function (file) {
  var input = file.target;

  var fr = new FileReader();

  fr.onload = function () {
    var dataURL = fr.result;
    var inputImage = document.getElementById('inputImage');
    inputImage.src = dataURL;
    img.src = fr.result;
  };
  fr.readAsDataURL(input.files[0]);

  setTimeout(poseEstimationProcess, 1000)
};

async function poseEstimationProcess() {
  // 1. PREPROCESSING
  let [resizedImageTensor, imageTensor] = preProcess(inputImage);

  // 2. INFERENCE
  let keypointsWithScores = await runInference(resizedImageTensor);

  // 3. DRAW PREDICTION
  drawPrediction(keypointsWithScores)
}


function preProcess(imgData) {

  console.log("Preprocessing...")

  // Convert to tensor
  let imageTensor = tf.browser.fromPixels(imgData, numChannels = 3)

  // Resize to 192 x 192
  let resized = tf.image.resizeBilinear(imageTensor, [192, 192])

  // Add a dimension to get a batch shape 
  let batched = resized.expandDims(0).toInt()

  return [batched, imageTensor]
}


async function runInference(resizedImageTensor) {
  console.log("Loading model...")
  let model = await tf.loadGraphModel('/pose_estimation_model/model.json');

  console.log("Predicting...")
  let prediction = model.predict(resizedImageTensor);

  return prediction;
}


const CONFIDENCE_THRESHOLD = 0.0;

const KEYPOINT_DICT = {
  0: 'nose',
  1: 'left_eye',
  2: 'right_eye',
  3: 'left_ear',
  4: 'right_ear',
  5: 'left_shoulder',
  6: 'right_shoulder',
  7: 'left_elbow',
  8: 'right_elbow',
  9: 'left_wrist',
  10: 'right_wrist',
  11: 'left_hip',
  12: 'right_hip',
  13: 'left_knee',
  14: 'right_knee',
  15: 'left_ankle',
  16: 'right_ankle'
}

function getKeypoints(keypointsWithScoresArray) {
  function _scaleCoordinate(coordinate, axis) {
    if (axis == "x") {
      return coordinate * img.width
    } else if (axis == "y") {
      return coordinate * img.height
    }
  }

  let x = []
  let y = []
  let text = []

  console.log("Scaling coordinates...")

  keypointsWithScoresArray.forEach(function (item, i) {
    x.push(_scaleCoordinate(item[1], "x"))
    y.push(_scaleCoordinate(item[0], "y"))
    text.push(KEYPOINT_DICT[i])
  });

  return [x, y, text]
}

async function convertTensorToArray(keypointsWithScores) {
  let keypointsWithScoresArray = await keypointsWithScores.array()
  return keypointsWithScoresArray[0][0]
}

const BONES = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [0, 5],
  [0, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 6],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16]
]

function getLines(dotsX, dotsY, text) {
  let lines = []

  for (let i = 0; i < BONES.length; i++) {
    let start = BONES[i][0]
    let end = BONES[i][1]

    if (dotsX[start] > 0.0 && dotsY[start] > 0.0 && dotsX[end] > 0.0 && dotsY[end] > 0.0) {
      let bone = {
        x: [dotsX[start], dotsX[end]],
        y: [dotsY[start], dotsY[end]],
        mode: 'lines',
        hovertemplate: '<extra></extra>'
      }
      lines.push(bone)
    }
  }

  return lines;
}


async function drawPrediction(keypointsWithScores) {
  console.log("Drawing prediction")

  Plotly.purge('outputImage') // Delete previous plot

  const keypointWithScoresArray = await convertTensorToArray(keypointsWithScores)

  const [dotsX, dotsY, texts] = getKeypoints(keypointWithScoresArray)

  let dots = {
    x: dotsX,
    y: dotsY,
    text: texts,
    mode: 'markers',
    type: 'scatter'
  }

  let data = [
    dots
  ]

  const lines = getLines(dotsX, dotsY, texts);
  data.push(...lines)

  const outputHeight = 500
  const outputWidth = (img.width * outputHeight) / img.height

  let layout = {
    autosize: false,
    showlegend: false,
    width: outputWidth,
    height: outputHeight,
    margin: { t: 0 },
    xaxis: {
      range: [0, img.width],
      showgrid: false
    },
    yaxis: {
      range: [img.height, 0],
      scaleranchor: "x",
      scaleratio: 1,
      showgrid: false
    },
    images: [
      {
        "source": `${document.getElementById('inputImage').src}`,
        "xref": "x",
        "yref": "y",
        "xanchor": "left",
        "yanchor": "bottom",
        "x": 0,
        "y": img.height,
        "sizex": img.width,
        "sizey": img.height,
        "sizing": "stretch",
        "opacity": 1.0,
        "layer": "below"
      }
    ]
  }

  let outputImagePlot = document.getElementById('outputImage');

  Plotly.plot(outputImagePlot, data, layout);
}






