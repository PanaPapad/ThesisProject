
using System.ComponentModel.DataAnnotations.Schema;

[Table("ProcessedData")]
public class ProcessedData{
    public long Id { get; set; }
    public long RawDataId { get; set; }
    public DateTime InsertDate { get; set; }
    public byte[] Data { get; set; }

    public ProcessedData(DateTime insertDate, byte[] data){
        InsertDate = insertDate;
        Data = data;
    }
}