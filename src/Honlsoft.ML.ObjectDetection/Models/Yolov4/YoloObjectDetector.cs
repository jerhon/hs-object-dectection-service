using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace Honlsoft.ML.ObjectDetection.Models.Yolov4
{
    public class YoloObjectDetector: IObjectDetector
    {
        private PredictionEngine<InputImage, YoloOutput> _predictionEngine;
        private YoloPostProcessor _postProcessor;
        
        public YoloObjectDetector( YoloPostProcessor postProcessor, YoloPredictionEngineFactory factory)
        {
            _predictionEngine = factory.BuildPredictionEngine();
            _postProcessor = postProcessor;
        }

        public float ScoreThreshold { get; set; } = 0.3f;
        public float IouThreshold { get; set; } = 0.7f;

        /// <summary>
        /// Detects objects via the Yolov4 Model.
        /// </summary>
        /// <param name="imageData"></param>
        /// <returns></returns>
        public IEnumerable<ObjectDetectionResult> DetectObjects(byte[] imageData)
        {
            // Predict via the ML.Net model.
            var prediction = _predictionEngine.Predict(new InputImage(new Bitmap(Image.FromStream(new MemoryStream(imageData)))));
            
            // Run post processing to get the obect detection results.
            var results = _postProcessor.GetResults(prediction, ScoreThreshold, IouThreshold);

            return results;
        }
    }
}