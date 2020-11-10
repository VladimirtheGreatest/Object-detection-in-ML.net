using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    /// <summary>
    ///ImageNetData is the input image data class and has the following String fields:
    ///ImagePath contains the path where the image is stored.
    ///Label contains the name of the file.
    /// </summary>
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
    }
}
