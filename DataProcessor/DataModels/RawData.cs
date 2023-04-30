using System.ComponentModel.DataAnnotations.Schema;

[Table("raw_data")]
public class RawData{
    [Column("id")]
    public long Id { get; set; }
    [Column("insert_date")]
    public DateTime InsertDate { get; set; }
    [Column("data")]
    public byte[] Data { get; set; }

    public virtual List<ProcessedData> ProcessedData { get; set; }

    public RawData(DateTime insertDate, byte[] data){
        InsertDate = insertDate;
        Data = data;
        ProcessedData = new List<ProcessedData>();
    }
}