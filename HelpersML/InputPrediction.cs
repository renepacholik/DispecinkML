using Microsoft.ML.Data;

namespace HelpersML
{
    public class InputPrediction
    {
        [ColumnName("PredictedLabel")]
        public int UrgLabel { get; set; }

        [ColumnName("Score"), VectorType(2)]
        public float[] Scores { get; set; }

    }
}
