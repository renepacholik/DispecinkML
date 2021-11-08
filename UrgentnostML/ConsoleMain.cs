using System;
using AutoML;

namespace UrgentnostML
{
    class ConsoleMain
    {
        static void Main(string[] args)
        {
            //Zvolená akce
            string action = "";
            //Pozice trenéra
            int trainer = 1;
            //Počet iterací cyklu
            int i = 0;
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
                }
                if (args.Length==0 | i > 0)
                {
                    Console.WriteLine(startup);
                    action = Console.ReadLine();  
                }
                
                
                // Přepínač pro menu
                   switch (action)
                {
                    case "a":
                        AutoTrain.AutoTraining();
                        break;
                    case "t":
                        Predict.Train(trainer);
                        break;
                    case "n":
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
                        break;
                    case "p":
                        Predict.TestMessage();
                        break;
                    case "exit":
                        Console.WriteLine("Exiting program.");
                        break;
                    case "?":
                        Console.WriteLine(startup);
                        break;
                    default:
                        Console.WriteLine("Your selected action does not match with the options.");
                        break;
                }
                i++;
            } while (!action.Equals("exit"));

        }
    }
}
