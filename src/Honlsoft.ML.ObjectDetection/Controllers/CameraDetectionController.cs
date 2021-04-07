using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Net.Http;
using System.Net.Mime;
using Microsoft.Extensions.Options;

namespace Honlsoft.ML.ObjectDetection.Controllers
{
    [ApiController]
    [Route("camera")]
    public class CameraDetectionController : ControllerBase
    {
        private readonly IObjectDetector _detector;
        private readonly IOptions<SourceOptions> _options;
        
        public CameraDetectionController(IObjectDetector detector, IOptions<SourceOptions> options)
        {
            _detector = detector;
            _options = options;
        }

        private async Task<byte[]> GetImageSourceAsync()
        {
            var client = new HttpClient();
            var image = await client.GetByteArrayAsync(_options.Value.ImageSourceUrl);
            return image;
        }
        
        // GET
        [HttpGet]
        [Route("objects")]
        public async Task<IEnumerable<ObjectDetectionResult>> Objects()
        {
            var image = await GetImageSourceAsync();
            return _detector.DetectObjects(image);
        }

        [HttpGet]
        [Route("boxes")]
        public async Task<IActionResult> Boxes()
        {
            // TODO: Split this out a bit, change the code to draw boxes / label outside of this.
            
            var image = await GetImageSourceAsync();
            var results = _detector.DetectObjects(image);
            var bitmap = new Bitmap(Image.FromStream(new MemoryStream(image)));

            using var g = Graphics.FromImage(bitmap);
            foreach (var result in results)
            {
                var label = $"{result.Label} ({result.Confidence})";
                var f = new Font(FontFamily.GenericMonospace, 48);
                var labelRect = g.MeasureString(label, f);
                g.DrawRectangle(Pens.Red, result.Left, result.Top, result.Width, result.Height);
                g.FillRectangle(Brushes.Crimson, result.Left, result.Top, labelRect.Width, labelRect.Height);
                g.DrawString(label, f, Brushes.White, result.Left, result.Top);
            }

            using var outputStream = new MemoryStream();
            bitmap.Save(outputStream, ImageFormat.Jpeg);
            return new FileContentResult(outputStream.ToArray(), MediaTypeNames.Image.Jpeg);
        }
    }
}