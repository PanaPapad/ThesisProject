using System.ComponentModel.DataAnnotations.Schema;
namespace DataModels;
[Table("processed_data")]
public class ProcessedData{
    [Column("id")]
    public long Id { get; set; }
    [Column("raw_data_id")]
    public long RawDataId { get; set; }
    [Column("insert_date")]
    public DateTime InsertDate { get; set; }
    [Column("data")]
    public byte[] Data { get; set; }
    public virtual RawData? RawData { get; set; }
    public virtual List<ResultsData> ResultsData { get; set; }
    public ProcessedData(byte[] data){
        InsertDate = DateTime.Now;
        Data = data;
        ResultsData = new List<ResultsData>();
    }
}