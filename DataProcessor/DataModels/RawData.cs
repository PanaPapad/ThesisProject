//This class defines a RawData record in the IDS database.

using System;
using System.Collections.Generic;

[Table("RawData")]
public class RawData{
    public long Id { get; set; }
    public DateTime InsertDate { get; set; }
    public byte[] Data { get; set; }
}