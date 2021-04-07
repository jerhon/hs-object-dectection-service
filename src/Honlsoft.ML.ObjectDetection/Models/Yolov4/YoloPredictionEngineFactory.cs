using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

namespace Honlsoft.ML.ObjectDetection.Models.Yolov4
{
    /// <summary>
    /// Creates a Yolo prediction engine factory.
    /// </summary>
    public class YoloPredictionEngineFactory
    {
        private readonly string _yoloModelPath;

        public YoloPredictionEngineFactory(IOptions<YoloOptions> options)
        {
            _yoloModelPath = options.Value?.ModelPath ?? GetDefaultModelPath();
        }

        private string GetDefaultModelPath() => Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "yolov4.onnx");

        /// <summary>
        /// Creates a ML.Net prediction engine for the Yolo based model.
        /// </summary>
        public Microsoft.ML.PredictionEngine<InputImage, YoloOutput> BuildPredictionEngine()
        {
            var outputColumnNames = new[]
            {
                "Identity:0",
                "Identity_1:0",
                "Identity_2:0",
            };
            var shapeDictionary = new Dictionary<string, int[]>()
            {
                {"input_1:0", new[] {1, 416, 416, 3}},
                {"Identity:0", new[] {1, 52, 52, 3, 85}},
                {"Identity_1:0", new[] {1, 26, 26, 3, 85}},
                {"Identity_2:0", new[] {1, 13, 13, 3, 85}},
            };
            var inputColumns = new[]
            {
                "input_1:0"
            };

            var mlContext = new MLContext();

            // Images captured need to be 416 x 416 pixels
            // Images need to be normalized to a good pixel format
            //  scale image changes the format of the color to a float from 0-1.0, it doesn't actually scale the image
            //  interleave pixel colors makes all the colors from one pixel next to each other.
            var pipeline = mlContext.Transforms.ResizeImages(
                    inputColumnName: "bitmap",
                    outputColumnName: "input_1:0",
                    imageWidth: 416,
                    imageHeight: 416,
                    resizing: ImageResizingEstimator.ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input_1:0",
                    scaleImage: 1f / 255f,
                    interleavePixelColors: true))
                .Append(mlContext.Transforms.ApplyOnnxModel(outputColumnNames, inputColumns, _yoloModelPath, shapeDictionary));

            // Model is pretrained, so dodn't need to fit it with any data, hence the empty list.
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<InputImage>()));
            
            // creates the actual engine used for the model.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InputImage, YoloOutput>(model);
            return predictionEngine;
        }
    }
}
