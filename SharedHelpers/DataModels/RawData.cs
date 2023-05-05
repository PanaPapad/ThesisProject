using System.ComponentModel.DataAnnotations.Schema;
namespace DataModels;
[Table("raw_data")]
public class RawData{
    [Column("id")]
    public long Id { get; set; }
    [Column("insert_date")]
    public DateTime InsertDate { get; set; }
    [Column("data")]
    public byte[] Data { get; set; }

    public virtual List<ProcessedData> ProcessedData { get; set; }

    public RawData(byte[] data){
        Data = data;
        InsertDate = DateTime.Now;
        ProcessedData = new List<ProcessedData>();
    }
}