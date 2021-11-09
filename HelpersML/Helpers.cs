using System;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using Microsoft.ML;
using Serilog;

namespace HelpersML
{
    public static class Helpers
    {
        static ILogger logger;

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

        public static void OutputMultiClassMetricsToLog(ITransformer model, IDataView data, MLContext mlContext)
        {
            var dataView = model.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(dataView);
            var confusionTable = metrics.ConfusionMatrix.GetFormattedConfusionTable();
            Log.Information("|Metriky modelu|\n"+
                            $"Micro accuracy: {metrics.MicroAccuracy}\n"+
                            $"Macro accuracy: {metrics.MacroAccuracy}\n"+
                            $"Log Loss: {metrics.LogLoss}\n" +
                            $"Log Loss reduction: {metrics.LogLossReduction}\n" +
                            confusionTable);
           
        }


        //Určí urgentnost a vypíše jí společně s pravděpodobností každé úrovně
        public static void DetermineUrgentnost(PredictionEngine<Input, InputPrediction> predictor, string message, bool cmd) 
        {
            var input = new Input { OrigText = message };

            InputPrediction prediction;
         
            prediction = predictor.Predict(input);
            
            string text = "";

            if (prediction.UrgLabel == 0) 
             {
                 text +="0;";
             }
             else if (prediction.UrgLabel == 1) 
             {
                 text += "1;";
             }
             else if (prediction.UrgLabel == 2) 
             {
                 text += "2;";
             }

            
             text += $"'2': {prediction.Scores[0]:0.000}, '1': {prediction.Scores[1]:0.000}, '0': {prediction.Scores[2]:0.000}";

            if (cmd == true)
            {
                File.WriteAllText(System.Configuration.ConfigurationManager.AppSettings["predictPath"], text);
            }
            else
            {
                Console.WriteLine(text);
            }
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

        //Aktualizuje a uloží hodnoty v App.config souboru
        public static void UpdateConfig(string key, string value)
        {
            Configuration conf = ConfigurationManager.OpenExeConfiguration(ConfigurationUserLevel.None);
            conf.AppSettings.Settings[key].Value = value;
            conf.Save();

            ConfigurationManager.RefreshSection("appSettings");
        }
    }
}
