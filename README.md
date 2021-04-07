# Overview

This application monitors a camera feed via an HTTP or HTTPS URL.
It will return the current set of objects detected in a picture as JSON and it has a seperate endpoint to take the JPEG from the camera feed and draw boxes around the objects.

Many of the projects I've seen on the web are of taking a set of static pictures and performing object detection.
I wanted to hook this up to a camera to see things in semi-real time.

## Design

I have a separate .NET project which takes pictures from my Raspberry PI and serves them via HTTP.
Whenever this project gets a request to look at the latest image, it will retrieve it from the Raspberry PI, and run an object detection ML model against it.
It can then use that to return a JSON of the objects in view of the camera via JSON, or a picture with bounding boxes.
The application uses a pre-trained YOLOv4 model to detect objects.

## Running The Application

Runs like a normal Asp.NET Core application.  Use "dotnet run" in the src\Honlsoft.ML.ObjectDetection project

### Yolov4 Model

yolo4.onnx needs to be downloaded and added to the root of the repository.
It's rather large so I didn't want to keep it in my git repo.

https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4/model

### Appsettings.json

Add a source URL.

```json
{
  "Source": {
    "ImageSourceUrl": "http url for image feed"
  }
}
```

## Credits

The code for the processing via the Yolo v4 model came from: https://github.com/BobLd/YOLOv4MLNet which was ported from the python code for the Yolo v4 model.
I changed just a few minor details to make it easier to consume.

Yolov4 Model Information: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4

## TODO

* The post processing code isn't entirely complete, there were some TODOs left there.
