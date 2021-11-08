using HelpersML;
using Microsoft.ML;
using System;
using System.IO;

namespace UrgentnostML
{
    class Predict
    {
        private static DataViewSchema dataSchema;
        private static MLContext mlContext;
        public static void Train(int trainerPosition) {

            mlContext = new MLContext();
            //Načtení dat za pomocí schéma Input a určení oddělovače
            var data = mlContext.Data.LoadFromTextFile<Input>(System.Configuration.ConfigurationManager.AppSettings["dataPath"], hasHeader: true, separatorChar: '\t');
            //Vytvoření pipeline a předpřipravení dat, reprezentujeme text jako vektor čísel pro ML
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", "OrigText"))
                .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"));

            Console.WriteLine("How many iterations would you like to go through?\n(Enter a number of iterations or press \"Enter\" for a default value of 100 iterations.)");

            //Volba počtu iterací
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

            //Vytvoření výchozího trenéra
            var trainer = mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features", numberOfIterations: iterations)
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            //Volba trenéra
            switch (trainerPosition)
            {
                case 1:
                    Console.WriteLine("Using the default trainer LightGbm");
                    break;
                case 2:
                    trainer = mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.AveragedPerceptron(numberOfIterations: iterations))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                    Console.WriteLine("Using the selcted trainer AveragedPerceptron");
                    break;
                case 3:
                    trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features", maximumNumberOfIterations: iterations)
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                    Console.WriteLine("Using the selcted trainer SdcaMaximumEntropy");
                    break;
                case 4:
                    trainer = mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression("Label", "Features", numberOfIterations: iterations))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                    Console.WriteLine("Using the selcted trainer SymbolicSgdLogisticRegression");
                    break;
                case 5:
                    trainer = mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.LinearSvm("Label", "Features", numberOfIterations: iterations))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                    Console.WriteLine("Using the selcted trainer LinearSvm");
                    break;
                case 6:
                    trainer = mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SgdCalibrated("Label", "Features", numberOfIterations: iterations))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                    Console.WriteLine("Using the selcted trainer SgdCalibrated");
                    break;
                case 7:
                    trainer = mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SgdNonCalibrated("Label", "Features", numberOfIterations: iterations))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                    Console.WriteLine("Using the selcted trainer SgdNonCalibrated");
                    break;
                default:
                    Console.WriteLine("Using the default trainer LightGbm");
                    break;

            }

            
                
            //Vložení trenéra do pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Training the model..");

            ITransformer model;
            //Spuštění trénování s daty
            using (new Helpers.PerformanceTimer("Training of the model"))
            {
                model = trainingPipeline.Fit(data);
            }
            //Uložení modelu do zip douboru
            mlContext.Model.Save(model, data.Schema, Path.Combine($"{Environment.CurrentDirectory}", "..", "..", "..", "UrgentnostMLModel.zip"));

            //Vypíše všechny metriky trenéra
            Helpers.OutputMultiClassMetrics(model, data, mlContext);

            
            
        }

        //Zjistí úroveň urgentnosti vložené zprávy
        public static void TestMessage() 
        {
            mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<Input>(System.Configuration.ConfigurationManager.AppSettings["dataPath"], hasHeader: true, separatorChar: '\t');

            dataSchema = data.Schema;
            ITransformer model = mlContext.Model.Load(System.Configuration.ConfigurationManager.AppSettings["modelPath"],  out dataSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<Input, InputPrediction>(model);

            Console.WriteLine("\nType a message to be checked or type \"exit\" to return to the menu.");
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
