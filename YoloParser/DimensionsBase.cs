using System;
using System.Collections.Generic;
using System.Text;

namespace ObjectDetection.YoloParser
{    /// <summary>
     ///  X contains the position of the object along the x-axis.
     /// Y contains the position of the object along the y-axis.
     /// Height contains the height of the object.
     /// Width contains the width of the object.
     /// </summary> 
    public class DimensionsBase
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
    }
}
