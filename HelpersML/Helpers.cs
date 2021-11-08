using System;
using System.Diagnostics;
using Microsoft.ML;

namespace HelpersML
{
    public static class Helpers
    {

        //Vypíše matici záměn obsahující počet správných a špatných předpovědí a metriky modelu
        public static void OutputMultiClassMetrics(ITransformer model, IDataView data, MLContext mlContext)
        {
            var dataView = model.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(dataView);
            var confusionTable = metrics.ConfusionMatrix.GetFormattedConfusionTable();
            Console.WriteLine($"\nMicro accuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"Macro accuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");
            Console.WriteLine($"Log Loss reduction: {metrics.LogLossReduction}");
            Console.Write(confusionTable);
        }


        //Určí urgentnost a vypíše jí společně s pravděpodobností každé úrovně
        public static void DetermineUrgentnost(PredictionEngine<Input, InputPrediction> predictor, string message) 
        {
            var input = new Input { OrigText = message };

            InputPrediction prediction;

            using (new PerformanceTimer("Prediction", true)) 
            {
                prediction = predictor.Predict(input);
            }
            Console.Write($"The message '{input.OrigText}' is classified as ");
            if (prediction.UrgLabel == 0) 
            {
                Console.Write("'0'");
            }
            else if (prediction.UrgLabel == 1) 
            {
                Console.Write("'1'");
            }
            else if (prediction.UrgLabel == 2) 
            {
                Console.Write("'2'");
            }
            Console.Write($" with precision: '2': {prediction.Scores[0]:0.000}, '1': {prediction.Scores[1]:0.000}, '0': {prediction.Scores[2]:0.000}\n");
        }


        //Měří čas
        public class PerformanceTimer : IDisposable
        {
            private readonly string _message;
            private readonly bool _showTicks;
            private readonly Stopwatch _timer;

            public PerformanceTimer(string message, bool showTicks = false)
            {
                _message = message;
                _showTicks = showTicks;
                _timer = new Stopwatch();
                _timer.Start();
            }

            public void Dispose()
            {
                _timer.Stop();
                Console.WriteLine($"{_message} took {_timer.ElapsedMilliseconds}ms{(_showTicks ? $" ({_timer.ElapsedTicks} ticks)" : string.Empty)}.");
            }
        }
    }
}
