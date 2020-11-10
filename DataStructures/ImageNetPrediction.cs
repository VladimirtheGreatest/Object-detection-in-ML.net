using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    /// <summary>
    /// PredictedLabel contains the dimensions, 
    /// objectness score, and class probabilities 
    /// for each of the bounding boxes detected in an image.
    /// </summary>
    public class ImageNetPrediction
    {      
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
