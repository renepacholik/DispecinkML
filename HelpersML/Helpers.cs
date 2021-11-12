using System;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using Microsoft.ML;
using Serilog;
using Newtonsoft.Json;
using System.Collections.Generic;

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
            try
            {
                var dataView = model.Transform(data);
                var metrics = mlContext.MulticlassClassification.Evaluate(dataView);
                var confusionTable = metrics.ConfusionMatrix.GetFormattedConfusionTable();
                Log.Information("|Metriky modelu|\n" +
                                $"Micro accuracy: {metrics.MicroAccuracy}\n" +
                                $"Macro accuracy: {metrics.MacroAccuracy}\n" +
                                $"Log Loss: {metrics.LogLoss}\n" +
                                $"Log Loss reduction: {metrics.LogLossReduction}\n" +
                                confusionTable);
            }
            catch (Exception e)
            {
                Log.Error("An unexpected error has occurred:\n" + e.Message + "\n" + e.StackTrace);
            }

        } 


        //Určí urgentnost a vypíše jí společně s pravděpodobností každé úrovně
        public static void DetermineUrgentnost(PredictionEngine<Input, InputPrediction> predictor, string message, bool cmd, string path) 
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
            try
            {
                if (cmd == true)
                {
                    string end = System.Configuration.ConfigurationManager.AppSettings["predictFileType"];
                    //Kontrola cesty k uložení předpovědi
                    if (!File.Exists(path) & Directory.Exists(path))
                    {
                        path =  Path.Combine(path, "out"+end);
                    }
                    else if (!File.Exists(path))
                    {
                        Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), "Prediction"));
                        path = Path.Combine(Directory.GetCurrentDirectory(), "Prediction", "out"+end);
                    }
                    
                    if (end.Equals(".txt"))
                    {
                        //Zapsání předpovědi do txt souboru
                        File.WriteAllText(path, text);
                        Log.Information("The prediction has been written to " + path + " with file type " + end);
                    }
                    else if (end.Equals(".json"))
                    {
                        string[] textArr = text.Split(";");
                        List<JsonHelper.predictData> data = new List<JsonHelper.predictData>();
                        data.Add(new JsonHelper.predictData()
                        {
                            prediction = int.Parse(textArr[0]),
                            percentage0 = Math.Round(prediction.Scores[2], 3),
                            percentage1 = Math.Round(prediction.Scores[1], 3),
                            percentage2 = Math.Round(prediction.Scores[0], 3)

                        });

                        string json = JsonConvert.SerializeObject(data.ToArray());
                        
                        File.WriteAllText(path, json);
                        Log.Information("The prediction has been written to " + path + " with file type " + end);
                    }
                    else
                    {
                        Log.Error("There wasn't found a valid data type in config file at value predictFileType.");
                    }
                    
                    
                }
                else
                {
                    Console.WriteLine(text);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.Error("The file was not found when PREDICTING A MESSAGE:\n" +e.Message+"\n"+ e.StackTrace);
            }
            catch (Exception e)
            {
                Log.Error("An unexpected error has occurred:\n" + e.Message + "\n"+ e.StackTrace);
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
