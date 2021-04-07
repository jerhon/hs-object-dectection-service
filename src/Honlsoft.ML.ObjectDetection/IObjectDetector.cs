using System.Collections.Generic;

namespace Honlsoft.ML.ObjectDetection
{
    public record ObjectDetectionResult(float Left, float Top,  float Right, float Bottom, string Label, float Confidence)
    {
        public float Width => Right - Left;
        public float Height => Bottom - Top;

    }
    
    public interface IObjectDetector
    {
        IEnumerable<ObjectDetectionResult> DetectObjects(byte[] imageData);
    }
}