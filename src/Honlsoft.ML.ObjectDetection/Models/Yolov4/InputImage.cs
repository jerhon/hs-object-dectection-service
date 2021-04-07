using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Honlsoft.ML.ObjectDetection.Models.Yolov4
{
    /// <summary>
    /// Input for the Yolo v4 model.
    /// </summary>
    public class InputImage
    {
        public InputImage(Bitmap image)
        {
            Image = image;
        }

        [ColumnName("bitmap")]
        [ImageType(416, 416)]
        public Bitmap Image { get; }

        [ColumnName("width")] public float ImageWidth => Image.Width;

        [ColumnName("height")] public float ImageHeight => Image.Height;
    }
}