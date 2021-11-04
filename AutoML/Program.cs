using HelpersML;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace AutoML
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<Input>(Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "Input", "extracted_csv"), hasHeader: true, separatorChar: '\t');//, hasHeader: true, separatorChar: ','

            Console.WriteLine("For how long would you like to search for the best trainer?\n(Enter a number in seconds or press \"Enter\" for a default value of 45s)");
           
            String input = Console.ReadLine();
            if (uint.TryParse(input, out uint resultS)) 
            {
                Console.WriteLine("Starting the experiment...\nEstimated time of the experiment "+resultS+"s");
            }
            else
            {
                resultS = 45;
                Console.WriteLine("Starting the experiment...\nEstimated time of the experiment using a default value " + resultS+"s");
            }
            
            var settings = new MulticlassExperimentSettings()
                {
                    OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
                    MaxExperimentTimeInSeconds = resultS
                };
                settings.Trainers.Remove(MulticlassClassificationTrainer.FastTreeOva);
                settings.Trainers.Remove(MulticlassClassificationTrainer.FastForestOva);

                Console.WriteLine("Starting the experiment");
                var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(settings);

                var progress = new Progress<RunDetail<MulticlassClassificationMetrics>>(detail =>
                {
                    if (detail.ValidationMetrics != null)
                    {
                        Console.WriteLine($"\n||Model||: {detail.TrainerName}");
                        Helpers.OutputMultiClassMetrics(detail.Model, data, mlContext);
                        Console.WriteLine($"\n|Time|: {detail.RuntimeInSeconds:###0.000}s\n");
                    }
                });
                var result = experiment.Execute(data, labelColumnName: "Label", progressHandler: progress);

                Console.WriteLine($"Winner: {result.BestRun.TrainerName}");

                Helpers.OutputMultiClassMetrics(result.BestRun.Model, data, mlContext);

            }
        }
    }

