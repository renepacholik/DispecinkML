using Microsoft.ML.Data;


namespace HelpersML
{

    //Schéma pro vstupní data
    public class Input
    {
        [LoadColumn(0), ColumnName("Label")]
        public int Urgentnost { get; set; }

        [LoadColumn(1)]
        public string OrigText { get; set; }
    }
}
