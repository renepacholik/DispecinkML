using System;
using Microsoft.ML.Data;

namespace HelpersML
{
    public class InputPrediction
    {
        [ColumnName("PredictedLabel")]
        public int UrgLabel { get; set; }

       
    }
}
