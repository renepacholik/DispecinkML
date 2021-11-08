using HelpersML;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;

namespace AutoML
{
    public static class AutoTrain
    {
        public static void AutoTraining()
        {
            var mlContext = new MLContext();

            //Načtení dat za pomocí schéma Input a určení oddělovače
            var data = mlContext.Data.LoadFromTextFile<Input>(System.Configuration.ConfigurationManager.AppSettings["dataPath"], hasHeader: true, separatorChar: '\t');

            Console.WriteLine("For how long would you like to search for the best trainer?\n(Enter a number in seconds or press \"Enter\" for a default value of 60s)");
           

            //Volba délky experimentu
            String input = Console.ReadLine();
            if (uint.TryParse(input, out uint resultS)) 
            {
                Console.WriteLine("Starting the experiment...\nEstimated time of the experiment "+resultS+"s");
            }
            else
            {
                resultS = 60;
                Console.WriteLine("Starting the experiment...\nEstimated time of the experiment using a default value " + resultS+"s");
            }

            /*Vytvoření nastavení
                OptimizingMetric vybýrá metriku podle které má vybrat nejlepšího trenéra
                MaxExperimentTimeInSeconds nastaví čas experimentu*/
            var settings = new MulticlassExperimentSettings()
                {
                    OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
                    MaxExperimentTimeInSeconds = resultS
                };

            //Odstranění trenérů, protože se nehodí pro naše data
                settings.Trainers.Remove(MulticlassClassificationTrainer.FastTreeOva);
                settings.Trainers.Remove(MulticlassClassificationTrainer.FastForestOva);
                settings.Trainers.Remove(MulticlassClassificationTrainer.LbfgsLogisticRegressionOva);

            //Vytvoření automatického hledání za pomocí nastavení
                var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(settings);

            //Vytvoření vypisování průběhu testování
                var progress = new Progress<RunDetail<MulticlassClassificationMetrics>>(detail =>
                {
                    if (detail.ValidationMetrics != null)
                    {
                        Console.WriteLine($"\n||Model||: {detail.TrainerName}");
                        Helpers.OutputMultiClassMetrics(detail.Model, data, mlContext);
                        Console.WriteLine($"\n|Time|: {detail.RuntimeInSeconds:###0.000}s\n");
                    }
                });

            //Spustí experiment
                var result = experiment.Execute(data, labelColumnName: "Label", progressHandler: progress);

            //Vypíše nejlepšího trenéra
                Console.WriteLine($"Winner: {result.BestRun.TrainerName}");

            //Vypíše všechny metriky nejlepšího trenéra
                Helpers.OutputMultiClassMetrics(result.BestRun.Model, data, mlContext);

            }
        }
    }

