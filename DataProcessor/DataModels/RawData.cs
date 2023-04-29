//This class defines a RawData record in the IDS database.

using System.ComponentModel.DataAnnotations.Schema;

[Table("RawData")]
public class RawData{
    public long Id { get; set; }
    public DateTime InsertDate { get; set; }
    public byte[] Data { get; set; }

    public RawData(DateTime insertDate, byte[] data){
        InsertDate = insertDate;
        Data = data;
    }
}