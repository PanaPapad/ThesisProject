using System.Diagnostics;
using System.Text;
using DataModels;
using IDSDatabaseTools;
using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using SharedHelpers.QueueTools;

namespace DataProcessor;
public class RawDataConsumer : EventingBasicConsumer
{
    private readonly DatabaseAccessor databaseAccessor;
    private readonly QueueMessagerService queueMessagerService;
    private readonly string cfmPath;
    private readonly string  outputDirectory;
    private readonly string nextQueueId;
    private readonly string failedQueueId;
    public RawDataConsumer(IModel channel, DatabaseAccessor accessor, QueueMessagerService qms, IConfiguration settings) : base(channel)
    {
        this.databaseAccessor = accessor;
        this.queueMessagerService = qms;
        this.cfmPath = settings["cfmPath"] ?? throw new ArgumentNullException("CicFlowMeterPath is null");
        this.outputDirectory = settings["outputDirectory"] ?? throw new ArgumentNullException("CicFlowMeterOutputDirectory is null");
        this.nextQueueId = settings["nextQueueId"] ?? throw new ArgumentNullException("NextQueueId is null");
        this.failedQueueId = settings["failedQueueId"] ?? throw new ArgumentNullException("FailedQueueId is null");
        this.Received += async (model, ea) =>
                {
                    var body = ea.Body.ToArray();
                    var message = Encoding.UTF8.GetString(body);
                    Console.WriteLine(" [x] Received {0}", message);

                    // Process the pcap file
                    await ProcessRawDataTask(message);

                    // Acknowledge the message
                    channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
                };
    }

    private async Task ProcessRawDataTask(string message)
    {
        long rawDataId = long.Parse(message);
        // Get the pcap file from the database using the id in the message
        var pcapFile = GetPcapFile(message);

        // Process the pcap file using CicFlowMeter
        byte[] processedPcapFile = ProcessPcapFile(pcapFile);
        //if the processed pcap file is null then send the message to the failed queue
        if(processedPcapFile == null){
            //Write the pcap file to the failed queue
            queueMessagerService.SendMessage("Failed to process RawData with id: " + message, failedQueueId);
            throw new NullReferenceException("Processed pcap file is null");
        }

        // Save the processed csv file to the db and send a message to the next queue
        var savedRecord = await SaveProcessedPcapFile(processedPcapFile, rawDataId);
        if(!savedRecord){
            //Write the pcap file to the failed queue
            queueMessagerService.SendMessage(processedPcapFile.ToString() ?? "No data found from processed RawData with id: "+message, failedQueueId);
        }
    }

    private Task<bool> SaveProcessedPcapFile(byte[] processedPcapFile, long rawDataId)
    {
        long newId;
        try{
            newId = databaseAccessor.AddProcessedData(new ProcessedData(processedPcapFile, rawDataId));
        }
        catch(Exception e){
            Console.WriteLine("Error saving processed pcap file to database: " + e.Message);
            return Task.FromResult(false);
        }
        //Send a message to the next queue
        queueMessagerService.SendMessage(newId.ToString(), nextQueueId);
        return Task.FromResult(true);
    }

    private byte[] ProcessPcapFile(byte[] pcapFile)
    {
        //Create a temp file to store the pcap file data
        var tempFile = Path.GetTempFileName();
        //Change file extension to pcap
        tempFile = Path.ChangeExtension(tempFile, ".pcap");
        //Write the pcap file data to the temp file
        File.WriteAllBytes(tempFile, pcapFile);
        //Run the cfm bat file. The process should run in the .bat directory context
        var process = Process.Start(new ProcessStartInfo
        {
            FileName = "cmd.exe",
            Arguments = $"/c {cfmPath} {tempFile} {outputDirectory}",
            WorkingDirectory = Path.GetDirectoryName(cfmPath) ?? throw new NullReferenceException("Could not get working directory from cfmPath")
        }) ?? throw new NullReferenceException("Could not start process. Check cfmPath and outputDirectory");

        //var process = Process.Start(cfmPath,tempFile+ " " + outputDirectory);
        //Wait for the process to exit
        process.WaitForExit();
        //Get the csv file from the output directory
        var csvFile = Directory.GetFiles(outputDirectory).FirstOrDefault() ?? throw new NullReferenceException("Could not find csv file in output directory");
        //Read the csv file into a byte array
        var csvFileData = File.ReadAllBytes(csvFile);
        //Delete the temp file
        File.Delete(tempFile);
        //Delete the csv file
        File.Delete(csvFile);
        //Return the csv file data
        return csvFileData ?? throw new NullReferenceException("csvFileData is null");
    }

    private byte[] GetPcapFile(string message)
    {
        long id = long.TryParse(message, out id) ? id : 0;
        var record =  databaseAccessor.GetRawDataWithId(id) ?? throw new NullReferenceException("Could not find record with id " + id);
        return record.Data;
    }


}