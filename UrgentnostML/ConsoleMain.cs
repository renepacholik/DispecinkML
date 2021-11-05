using System;
using AutoML;

namespace UrgentnostML
{
    class ConsoleMain
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Choose action: 'a' - Finds the best trainer for your data.\n" +
                              "               't' - Trains on the data using the selected or default trainer.\n" +
                              "               'n' - Selects a new default trainer.\n" +
                              "               'p' - Preditcs the urgency of a message.\n" +
                              "            \"exit\" - Exits the program.\n");
            String action;
            do
            {
                 action = Console.ReadLine();

                switch (action)
                {
                    case "a":
                        AutoTrain.AutoTraining();
                        break;
                    case "t":
                        Predict.Train();
                        break;
                    case "n":
                        Console.WriteLine("TODO");
                        break;
                    case "p":
                        Predict.TestMessage();
                        break;
                    case "exit":
                        Console.WriteLine("Exiting program.");
                        break;
                    default:
                        Console.WriteLine("Your selected action does not match with the options.");
                        break;
                }

            } while (!action.Equals("exit"));

        }
    }
}
