namespace Honlsoft.ML.ObjectDetection.Models.Yolov4
{
    public class YoloConstants
    {
        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/anchors/yolov4_anchors.txt
        public static readonly float[][][] Anchors = {
            new [] { new [] { 12f, 16f },   new [] { 19f, 36f   }, new [] { 40f, 28f } },
            new [] { new [] { 36f, 75f },   new [] { 76f, 55f   }, new [] { 72f, 146f } },
            new [] { new [] { 142f, 110f }, new [] { 192f, 243f }, new [] { 459f, 401f } }
        };

        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/core/config.py#L18
        public static readonly float[] Strides = { 8, 16, 32 };

        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/core/config.py#L20
        public static readonly float[] XYScale = { 1.2f, 1.1f, 1.05f };

        public static readonly int[] Shapes = { 52, 26, 13 };
        
        public static readonly string[] ClassNames = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    }
}