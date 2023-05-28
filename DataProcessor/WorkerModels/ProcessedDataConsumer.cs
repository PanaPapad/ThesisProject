
using System.Diagnostics;
using System.Text;
using DataModels;
using IDSDatabaseTools;
using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using SharedHelpers.QueueTools;

namespace DataProcessor;
public class ProcessedDataConsumer : EventingBasicConsumer
{
    private readonly DatabaseAccessor databaseAccessor;
    private readonly QueueMessagerService queueMessagerService;
    private readonly ConsumerSettings settings;
    public ProcessedDataConsumer(IModel channel, DatabaseAccessor dba, QueueMessagerService qms, ConsumerSettings settings) : base(channel)
    {
        this.databaseAccessor = dba;
        this.queueMessagerService = qms;
        this.settings = settings;
        this.Received += async (model, ea) =>
                {
                    var body = ea.Body.ToArray();
                    var message = Encoding.UTF8.GetString(body);
                    Console.WriteLine(" [x] Received {0}", message);

                    // Process the csv file
                    await ProcessProcessedDataTask(message);

                    // Acknowledge the message
                    channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
                };
    }

    private async Task ProcessProcessedDataTask(string message)
    {
        long dataId = long.Parse(message);
        //Get data from db
        var data = GetCSVData(dataId);
        byte[] resultData = GetResultData(data);
        //Write ids output to db
        var savedRecord = await SaveIdsOutput(resultData, dataId);
        if(!savedRecord){
            //Write the csv file to the failed queue
            queueMessagerService.SendMessage(resultData.ToString() ?? "No data found from ProcessedData with id: "+message, settings.FailedQueueId);
        }
    }

    private Task<bool> SaveIdsOutput(byte[] resultData, long dataId)
    {
        long newId;
        try{
            newId = databaseAccessor.AddResultsData(new ResultsData(resultData, dataId));
        }catch(Exception e){
            Console.WriteLine("Error saving ids output to database: " + e.Message);
            return Task.FromResult(false);
        }
        //Send message to next queue
        queueMessagerService.SendMessage(newId.ToString(), settings.OutputQueueId);
        return Task.FromResult(true);
    }

    private byte[] GetResultData(byte[] data)
    {
        //Write data to temp csv file
        var tempFile = Path.GetTempFileName();
        tempFile = Path.ChangeExtension(tempFile, ".csv");
        File.WriteAllBytes(tempFile, data);
        //Run ids on temp csv file
        var execSettings = settings.Settings;
        Process process;
        if (execSettings.BaseCommand.Equals("EXEC"))
        {
            process = Process.Start(new ProcessStartInfo
            {
                FileName = execSettings.ExecutablePath,
                Arguments = $"{tempFile} {execSettings.OutputPath} " + execSettings.ExecutableArguments,
                WorkingDirectory = Path.GetDirectoryName(execSettings.ExecutablePath) ?? throw new NullReferenceException("Could not get working directory from processorPath")
            }) ?? throw new NullReferenceException("Could not start process. Check ExecutablePath and outputDirectory");
        }
        else
        {
            process = Process.Start(new ProcessStartInfo
            {
                FileName = execSettings.BaseCommand,
                Arguments = $"/c {execSettings.ExecutablePath} {tempFile} {execSettings.OutputPath} " + execSettings.ExecutableArguments,
                WorkingDirectory = Path.GetDirectoryName(execSettings.ExecutablePath) ?? throw new NullReferenceException("Could not get working directory from processorPath")
            }) ?? throw new NullReferenceException("Could not start process. Check ExecutablePath and outputDirectory");
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

    private byte[] GetCSVData(long id){
        var record = databaseAccessor.GetProcessedDataWithId(id) ??
         throw new NullReferenceException("Processed Data record with id: " + id + "doesn't exist");
        return record.Data;
    }

    
}