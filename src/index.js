if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("sw.js")
      .then(reg => console.log("Service Worker: Registered (Pages)"))
      .catch(err => console.log(`Service Worker: Error: ${err}`));
  });
}

import "./styles/index.scss";
import * as tf from "@tensorflow/tfjs";
import yolo from "tfjs-yolo";

const loader = document.getElementById("loader");
const spinner = document.getElementById("spinner");
const webcam = document.getElementById("webcam");
const wrapper = document.getElementById("webcam-wrapper");
const rects = document.getElementById("rects");
const v3 = document.getElementById("v3");
const v1tiny = document.getElementById("v1tiny");
const v2tiny = document.getElementById("v2tiny");
const v3tiny = document.getElementById("v3tiny");
const iswebcame = document.getElementById("iswebcam");

let myYolo;
let selected;
var camon = false;
var webcamStream;
var canvas;
var img;
var ctx;
document.getElementById("inp").onchange = function(e) {
  img = new Image();
  img.onload = draw;
  img.onerror = failed;
  img.src = URL.createObjectURL(this.files[0]);
};
async function draw() {
  canvas = document.getElementById("canvas");
  canvas.height = canvas.width * (img.height / img.width);
  var oc = document.createElement("canvas"),
    octx = oc.getContext("2d");

  oc.width = img.width * 0.5;
  oc.height = img.height * 0.5;
  octx.drawImage(img, 0, 0, oc.width, oc.height);

  // step 2
  octx.drawImage(oc, 0, 0, oc.width * 0.5, oc.height * 0.5);

  // step 3, resize to final size
  ctx = canvas.getContext("2d");
  ctx.drawImage(
    oc,
    0,
    0,
    oc.width * 0.5,
    oc.height * 0.5,
    0,
    0,
    canvas.width,
    canvas.height
  );
  myYolo = await yolo.v2tiny();

  imgrun();
}

async function imgrun() {
  console.log("Start with tensors: " + tf.memory().numTensors);
  const boxes = await myYolo.predict(canvas, { scoreThreshold: 0.2 });
  console.log(boxes);
  boxes.map(box => {
    ctx.lineWidth = 2;
    ctx.fillStyle = "red";
    ctx.strokeStyle = "red";
    ctx.rect(box["left"], box["top"], box["width"], box["height"]);
    ctx.fillText(box["class"], box["left"] + 5, box["top"] + 10);
    ctx.stroke();
  });
  console.log("End with tensors: " + tf.memory().numTensors);
}

function failed() {
  console.error("The provided file couldn't be loaded as an Image media");
}

(async function main() {
  try {
    v3.addEventListener("click", () => load(v3));
    v1tiny.addEventListener("click", () => load(v1tiny));
    v2tiny.addEventListener("click", () => load(v2tiny));
    v3tiny.addEventListener("click", () => load(v3tiny));
    iswebcame.addEventListener("click", () => changeCam());
  } catch (e) {
    console.error(e);
  }
})();

async function changeCam() {
  camon = !camon;
  if (camon == true) {
    try {
      console.log("enable webcam");
      await setupWebCam();
      load(v1tiny);
      run();
    } catch (e) {
      console.error(e);
    }
  } else {
    try {
      console.log("disable webcam");
      stopWebcam();
    } catch (e) {
      console.error(e);
    }
  }
}

function stopWebcam() {
  webcamStream.getVideoTracks()[0].stop();
  webcamStream.getVideoTracks()[0] = null;
}

async function setupWebCam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "environment" }
    });
    window.stream = stream;
    webcam.srcObject = stream;
    webcamStream = stream;
  }
}

async function load(button) {
  if (button == v3) {
    var r = confirm(
      "Full YOLO inference is slow in browser and may take 30+ seconds to predict one picture. Do you want to continue?"
    );
    if (!r) {
      return;
    }
  }

  if (myYolo) {
    myYolo.dispose();
    myYolo = null;
  }

  rects.innerHTML = "";
  loader.style.display = "block";
  spinner.style.display = "block";
  setButtons(button);

  setTimeout(async () => {
    switch (button) {
      case v3:
        progress(60);
        myYolo = await yolo.v3();
        break;
      case v1tiny:
        progress(16);
        myYolo = await yolo.v1tiny();
        break;
      case v2tiny:
        progress(11);
        myYolo = await yolo.v2tiny();
        break;
      default:
        progress(9);
        myYolo = await yolo.v3tiny();
    }
  }, 200);
}

function setButtons(button) {
  v3.className = "";
  v1tiny.className = "";
  v2tiny.className = "";
  v3tiny.className = "";
  button.className = "selected";
  selected = button;
}

function progress(totalModel) {
  let cnt = 0;
  Promise.all = (all => {
    return function then(reqs) {
      if (reqs.length === totalModel && cnt < totalModel * 2)
        reqs.map(req => {
          return req.then(r => {
            loader.setAttribute(
              "percent",
              ((++cnt / totalModel) * 50).toFixed(1)
            );
            if (cnt === totalModel * 2) {
              loader.style.display = "none";
              spinner.style.display = "none";
              loader.setAttribute("percent", "0.0");
            }
          });
        });
      return all.apply(this, arguments);
    };
  })(Promise.all);
}

async function run() {
  let interval = 1;
  if (myYolo) {
    let threshold = 0.3;
    if (selected == v3tiny) threshold = 0.2;
    else if (selected == v3) interval = 10;
    await predict(threshold);
  }
  setTimeout(run, interval * 100);
}

async function predict(threshold) {
  console.log(`Start with ${tf.memory().numTensors} tensors`);

  const start = performance.now();
  const boxes = await myYolo.predict(webcam, { scoreThreshold: threshold });
  const end = performance.now();

  console.log(`Inference took ${end - start} ms`);
  console.log(`End with ${tf.memory().numTensors} tensors`);

  drawBoxes(boxes);
}

let colors = {};

function drawBoxes(boxes) {
  console.log(boxes);
  rects.innerHTML = "";

  const cw = webcam.clientWidth;
  const ch = webcam.clientHeight;
  const vw = webcam.videoWidth;
  const vh = webcam.videoHeight;

  const scaleW = cw / vw;
  const scaleH = ch / vh;

  wrapper.style.width = `${cw}px`;
  wrapper.style.height = `${ch}px`;

  boxes.map(box => {
    if (!(box["class"] in colors)) {
      colors[box["class"]] =
        "#" + Math.floor(Math.random() * 16777215).toString(16);
    }

    const rect = document.createElement("div");
    rect.className = "rect";
    rect.style.top = `${box["top"] * scaleH}px`;
    rect.style.left = `${box["left"] * scaleW}px`;
    rect.style.width = `${box["width"] * scaleW - 4}px`;
    rect.style.height = `${box["height"] * scaleH - 4}px`;
    rect.style.borderColor = colors[box["class"]];

    const text = document.createElement("div");
    text.className = "text";
    text.innerText = `${box["class"]} ${box["score"].toFixed(2)}`;
    text.style.color = colors[box["class"]];

    rect.appendChild(text);
    rects.appendChild(rect);
  });
}
