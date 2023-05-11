using System.ComponentModel.DataAnnotations.Schema;
namespace DataModels;
[Table("results_data")]
public class ResultsData{
    [Column("id")]
    public long Id { get; set; }
    [Column("processed_data_id")]
    public long ProcessedDataId { get; set; }
    [Column("insert_date")]
    public DateTime InsertDate { get; set; }
    [Column("data")]
    public byte[] Data { get; set; }
    public virtual ProcessedData? ProcessedData { get; set; }

    public ResultsData(byte[] data, long processedDataId){
        InsertDate = DateTime.Now;
        ProcessedDataId = processedDataId;
        Data = data;
    }
}