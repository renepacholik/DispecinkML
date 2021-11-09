﻿using System;
using AutoML;
using HelpersML;
using Serilog;

namespace UrgentnostML
{
    class ConsoleMain
    {
        static void Main(string[] args)
        {
            //Deklarace a inicializace logování
            Log.Logger = new LoggerConfiguration()
                .WriteTo.File(System.Configuration.ConfigurationManager.AppSettings["logPath"])
                .CreateLogger();
            Log.Information("|START|");
                
            //Zvolená akce
            string action = "";
            //Pozice trenéra
            int trainer = 1;
            //Počet iterací cyklu
            int i = 0;
            //Počet iterací
            int iterations = 100;
            //Zpráva na otestování
            string message = "";
            //Jestli byl program spuštěn z programové řádky
            bool cmd = false;
                
            //Menu
            string startup = "Choose action: 'a' - Finds the best trainer for your data.\n" +
                                  "               't' - Trains on the data using the selected or default trainer.\n" +
                                  "               'n' - Selects a new default trainer.\n" +
                                  "               'p' - Preditcs the urgency of a message.\n" +
                                  "               '?' - Shows this menu whenever you are not performing an action.\n" +
                                  "            \"exit\" - Exits the program.\n";

            

            do
            {
                //Pokud při spouštění z příkazové řádky bude zadán argument přeskočí se vypsaní menu a rovnou přejde do zvolené akce
                if (args.Length > 0)
                {
                    action = args[0];
                    cmd = true;
                    Log.Information("Program has been launched from the command line with arguments.");
                }
                if (cmd == false)
                {
                    Console.WriteLine(startup);
                }
                if (cmd== false | i > 0)
                {
                    action = Console.ReadLine();  
                }
                
                
                
                // Přepínač pro menu
                   switch (action)
                {
                    case "a":
                        Log.Information("|Starting task| Auto training");
                        AutoTrain.AutoTraining();
                        Log.Information("|Ending task| Auto training");
                        break;
                    case "t":
                        Log.Information("|Starting task| Training");
                        if (cmd == true)
                        {
                            //Kontrola existence argumentů a vykonaní přiřazených příkazů při spuštění z příkazové řádky s argumenty
                            if (args.Length >= 2)
                            {
                                Helpers.UpdateConfig("dataPath", args[1]);
                            }
                            if (args.Length >= 3) 
                            {
                                if (!args[2].Equals("d"))
                                {
                                    iterations = int.Parse(args[2]);
                                }
                            }
                            if (args.Length >= 4)
                            {
                                if (!args[3].Equals("d"))
                                {
                                    trainer = int.Parse(args[3]);
                                }
                            }
                            if (args.Length >= 5) 
                            {
                                if (!args[4].Equals("d"))
                                {
                                    Helpers.UpdateConfig("modelPathSave", args[4]);
                                }
                            }
                            Predict.Train(trainer, iterations, cmd);
                            action = "exit";
                        }
                        else
                        {
                            Predict.Train(trainer, iterations, cmd);
                        }
                        Log.Information("|Ending task| Training");
                        break;
                    case "n":
                        Log.Information("|Starting task| Trainer selection");
                        Console.WriteLine("Select a new trainer: '1' - LightGbm\n" +
                                          "                      '2' - AveragedPerceptron\n" +
                                          "                      '3' - SdcaMaximumEntropy\n" +
                                          "                      '4' - SymbolicSgdLogisticRegression\n" +
                                          "                      '5' - LinearSvm\n" +
                                          "                      '6' - SgdCalibrated\n" +
                                          "                      '7' - SgdNonCalibrated\n");
                        string input = Console.ReadLine();
                            if (int.TryParse(input, out int tr) && tr > 0 && tr < 10)
                            {
                            trainer = tr;
                            Console.WriteLine("New trainer selected.");
                            }
                            else
                            {
                            Console.WriteLine("Wrong input, using the default trainer.");
                            }
                        Log.Information("|Ending task| Trainer selection");
                        break;
                    case "p":
                        Log.Information("|Starting task| Prediction");
                        //Kontrola existence argumentů a vykonaní přiřazených příkazů při spuštění z příkazové řádky s argumenty
                        if (cmd == true)
                        {
                            if (args.Length >= 2)
                            {
                                message = args[1];
                            }
                            if (args.Length >= 3)
                            {
                                if (!args[2].Equals("d"))
                                {
                                    Helpers.UpdateConfig("predictPath", args[2]);
                                }
                            }
                            if (args.Length >= 4)
                            {
                                if (!args[3].Equals("d"))
                                {
                                    Helpers.UpdateConfig("modelPathLoad", args[3]);
                                }
                            }
                        
                            Predict.TestMessageFile(message);
                            action = "exit";
                        }
                        else
                        {
                            Predict.TestMessage();
                        }
                        Log.Information("|Ending task| Prediction");
                        break;
                    case "exit":
                        Console.WriteLine("Exiting program.");
                        break;
                    case "?":
                        Console.WriteLine(startup);
                        break;
                    default:
                        Log.Information("Selected action did not match the options");
                        Console.WriteLine("Your selected action does not match with the options.");
                        break;
                }
                i++;
            } while (!action.Equals("exit"));
            Log.Information("|END|");
            Log.CloseAndFlush();

        }
    }
}
