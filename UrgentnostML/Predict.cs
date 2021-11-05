using HelpersML;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace UrgentnostML
{
    class Predict
    {
        private static DataViewSchema dataSchema;
        private static MLContext mlContext;
        public static void Train() {

            mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<Input>(System.Configuration.ConfigurationManager.AppSettings["dataPath"], hasHeader: true, separatorChar: '\t');
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", "OrigText"))
                .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"));

            Console.WriteLine("How many iterations would you like to go through?\n(Enter a number of iterations or press \"Enter\" for a default value of 100 iterations.)");

            String input = Console.ReadLine();
            if (int.TryParse(input, out int iterations) && iterations > 0)
            {
                Console.WriteLine("Number of iterations selected: " + iterations);
            }
            else
            {
                iterations = 100;
                Console.WriteLine("Using the default number of iterations: " + iterations);
            }

            //var trainer = mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features", numberOfIterations: iterations)
            var trainer = mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features", numberOfIterations: iterations)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Training the model..");

            ITransformer model;
            using (new Helpers.PerformanceTimer("Training of the model"))
            {
                model = trainingPipeline.Fit(data);
            }

            mlContext.Model.Save(model, data.Schema, Path.Combine($"{Environment.CurrentDirectory}", "..", "..", "..", "UrgentnostMLModel.zip"));
            
            Helpers.OutputMultiClassMetrics(model, data, mlContext);

            
            
        }

        public static void TestMessage() 
        {
            mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<Input>(System.Configuration.ConfigurationManager.AppSettings["dataPath"], hasHeader: true, separatorChar: '\t');

            dataSchema = data.Schema;
            ITransformer model = mlContext.Model.Load(System.Configuration.ConfigurationManager.AppSettings["modelPath"],  out dataSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<Input, InputPrediction>(model);

            Console.WriteLine("\nType a message to be checked or type \"exit\" to return to the menu");
            String message;
            while (!(message = Console.ReadLine())?.Equals("exit", StringComparison.OrdinalIgnoreCase) ?? true)
            {
                if (string.IsNullOrWhiteSpace(message))
                    continue;
                Helpers.DetermineUrgentnost(predictor, message);
            }

        }
        
    }
}
