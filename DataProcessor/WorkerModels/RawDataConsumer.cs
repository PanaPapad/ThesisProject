using System.Diagnostics;
using System.Runtime.InteropServices;
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
    private readonly ConsumerSettings settings;
    public RawDataConsumer(IModel channel, DatabaseAccessor accessor, QueueMessagerService qms, ConsumerSettings settings) : base(channel)
    {
        this.databaseAccessor = accessor;
        this.queueMessagerService = qms;
        this.settings = settings;
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
        var pcapFile = GetPcapFile(rawDataId);

        // Process the pcap file using the processing script/program
        byte[] processedPcapFile = ProcessPcapFile(pcapFile);
        //if the processed pcap file is null then send the message to the failed queue
        if (processedPcapFile == null)
        {
            //Write the pcap file to the failed queue
            queueMessagerService.SendMessage("Failed to process RawData with id: " + message, settings.FailedQueueId);
            throw new NullReferenceException("Processed pcap file is null");
        }

        // Save the processed csv file to the db and send a message to the next queue
        var savedRecord = await SaveProcessedPcapFile(processedPcapFile, rawDataId);
        if (!savedRecord)
        {
            //Write the pcap file to the failed queue
            queueMessagerService.SendMessage(processedPcapFile.ToString() ?? "No data found from processed RawData with id: " + message, settings.FailedQueueId);
        }
    }

    /**
    <summary>
    Save the processed pcap file to the database.
    Returns true if the record was saved successfully.
    </summary>
    */
    private Task<bool> SaveProcessedPcapFile(byte[] processedPcapFile, long rawDataId)
    {
        long newId;
        try
        {
            newId = databaseAccessor.AddProcessedData(new ProcessedData(processedPcapFile, rawDataId));
        }
        catch (Exception e)
        {
            Console.WriteLine("Error saving processed pcap file to database: " + e.Message);
            return Task.FromResult(false);
        }
        //Send a message to the next queue
        queueMessagerService.SendMessage(newId.ToString(), settings.OutputQueueId);
        return Task.FromResult(true);
    }

    /**
    <summary>
    Process the pcap file using CicFlowMeter.
    Returns the csv file as a byte array.
    </summary>
    */
    private byte[] ProcessPcapFile(byte[] pcapFile)
    {
        //Create a temp file to store the pcap file data
        var tempFile = Path.GetTempFileName();
        //Change file extension to pcap
        tempFile = Path.ChangeExtension(tempFile, ".pcap");
        //Write the pcap file data to the temp file
        File.WriteAllBytes(tempFile, pcapFile);
        //Run the cfm bat file.
        //The process should run in the .bat file dir to have access to the required libs
        var execSettings = settings.Settings;
        Process process;
        if (execSettings.BaseCommand.Equals("EXEC"))
        {
            process = Process.Start(new ProcessStartInfo
            {
                FileName = execSettings.ExecutablePath,
                Arguments = $"{tempFile} {execSettings.OutputPath} " + execSettings.ExecutableArguments,
                WorkingDirectory = Path.GetDirectoryName(execSettings.ExecutablePath) ?? throw new NullReferenceException("Could not get working directory from processorPath")
            }) ?? throw new NullReferenceException("Could not start process. Check processorPath and outputDirectory");
        }
        else
        {
            process = Process.Start(new ProcessStartInfo
            {
                FileName = execSettings.BaseCommand,
                Arguments = $"/c {execSettings.ExecutablePath} {tempFile} {execSettings.OutputPath} " + execSettings.ExecutableArguments,
                WorkingDirectory = Path.GetDirectoryName(execSettings.ExecutablePath) ?? throw new NullReferenceException("Could not get working directory from processorPath")
            }) ?? throw new NullReferenceException("Could not start process. Check processorPath and outputDirectory");
        }
        //Wait for the process to exit
        process.WaitForExit();
        //Get the csv file from the output directory
        var csvFile = Directory.GetFiles(execSettings.OutputPath).FirstOrDefault() ?? throw new NullReferenceException("Could not find csv file in output directory");
        //Read the csv file into a byte array
        var csvFileData = File.ReadAllBytes(csvFile);
        //Delete the temp file
        File.Delete(tempFile);
        //Delete the csv file
        File.Delete(csvFile);
        //Return the csv file data
        return csvFileData ?? throw new NullReferenceException("csvFileData is null");
    }

    /**
    <summary>
    Get the pcap file from the database.
    </summary>
    */
    private byte[] GetPcapFile(long id)
    {
        var record = databaseAccessor.GetRawDataWithId(id) ?? throw new NullReferenceException("Could not find record with id " + id);
        return record.Data;
    }


}