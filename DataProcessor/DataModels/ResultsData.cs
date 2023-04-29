using System.ComponentModel.DataAnnotations.Schema;

[Table("ResultsData")]
public class ResultsData{
    public long Id { get; set; }
    public long ProcessedDataId { get; set; }
    public DateTime InsertDate { get; set; }
    public byte[] Data { get; set; }

    public ResultsData(DateTime insertDate, byte[] data){
        InsertDate = insertDate;
        Data = data;
    }
}