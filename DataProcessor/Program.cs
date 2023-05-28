using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static SharedHelpers.HelperFunctions;
using SharedHelpers.QueueTools;
using RabbitMQ.Client.Exceptions;
using IDSDatabaseTools;

namespace DataProcessor;
public class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Starting DataProcessor...");
        Console.WriteLine("Reading Configuration...");
        //Get project directory
        var projectDirectory = Directory.GetCurrentDirectory();
        //Build the configuration object
        var config = GetConfigFromJsonFile(projectDirectory + "\\appConfig.json");
        Console.WriteLine("Configuration read.");
        //Connect to the database
        var dbAccessor = ConnectDB(config.GetSection("DatabaseSettings"));
        //Connect to RabbitMQ
        IModel channel;
        RabbitMQSettings rmq;
        channel = DeclareQueues(config.GetSection("RabbitMQSettings") , out rmq);
        //Begin consuming messages
        StartConsumers(channel, config.GetSection("ConsumerSettings"), rmq, dbAccessor);
        //Wait for user input to exit
        Console.WriteLine("Press [enter] to exit.");
        Console.ReadLine();
    }

    
    ///<summary>
    ///Helper method to exit the program with a message.
    ///</summary>
    private static void ExitWithMessage(string message)
    {
        Console.WriteLine(message);
        //Wait for user input to exit
        Console.WriteLine("Press [enter] to exit.");
        Console.ReadLine();
        Environment.Exit(1);
    }
    
    /**
    <summary>
        Method to declare all queues in the queueDefinitions.json file.
        Queue definitions are stored in a json file to allow for easy modification
        Queues that cannot be declared will be skipped and an error message will be printed
        Failure to read the json file will cause the program to exit
    </summary>
    */
    [Obsolete] //Replaced by new method
    private static void DeclareQueues(IConfigurationSection RabbitMQSettings, IModel channel)
    {
        //Get the path to the json file containing the queue definitions
        string filePath = RabbitMQSettings["RabbitMQDefinitionsFilePath"] ?? "";
        if (filePath == "")
        {
            ExitWithMessage("RabbitMQDefinitionsFilePath not found in appsettings.json");
        }
        string jsonString = "";
        JArray queuesList = new JArray();
        JObject jsonData = new JObject();
        try
        {
            jsonString = File.ReadAllText(filePath);
            jsonData = GetNonNullValue<JObject>(JsonConvert.DeserializeObject(jsonString) as JObject);
            queuesList = (JArray)GetNonNullValue(jsonData["queueList"]);
        }
        catch (NullReferenceException e)
        {
            ExitWithMessage($"Error reading queue definitions file at {filePath}.\n{e.Message}");
        }
        foreach (JObject queue in queuesList)
        {
            try
            {
                channel.QueueDeclare(
                    GetNonNullValue(queue["name"]).ToString(),
                    GetNonNullValue(queue["durable"]).ToObject<bool>(),
                    GetNonNullValue(queue["exclusive"]).ToObject<bool>(),
                    GetNonNullValue(queue["autoDelete"]).ToObject<bool>(),
                    GetNonNullValue(queue["arguments"]).ToObject<IDictionary<string, object>>());
            }
            catch (NullReferenceException e)
            {
                int index = queuesList.IndexOf(queue);
                Console.WriteLine($"Error declaring queue at index {index}.");
                Console.WriteLine(e.StackTrace);
                continue;
            }
        }
    }
    /**
        <summary>
            Method to connect and declare queues on the RabbitMQ server. Connection variables and queue
            definitions are read from the provided configuration section.
        </summary>
    */
    private static IModel DeclareQueues(IConfigurationSection RabbitMQSettings, out RabbitMQSettings rmq)
    {
        Console.WriteLine("Connecting to RabbitMQ...");
        rmq = new RabbitMQSettings(RabbitMQSettings);
        //Check that the queue definitions are valid and more than 0.
        if (rmq.Queues == null || rmq.Queues.Count == 0)
        {
            ExitWithMessage("No queues defined in queueDefinitions.json");
        }
        IModel? channel = null;
        try{
            var factory = rmq.GetConnectionFactory();
            channel = factory.CreateModel();
            Console.WriteLine("Connected to RabbitMQ.");
            Console.WriteLine("Declaring queues...");       
            rmq.DeclareQueues(channel, true);
            Console.WriteLine("Queues declared.");
        }
        catch(BrokerUnreachableException e){
            ExitWithMessage($"Could not connect to RabbitMQ server.\n{e.Message}");
        }
        catch(OperationInterruptedException e){
            ExitWithMessage($"Could not declare queues.\n{e.Message}");
        }
        catch(Exception e){
            ExitWithMessage($"An unknown error occured.\n{e.Message}");
        }
        //Check that channel is open.
        if (channel==null || !channel.IsOpen)
        {
            ExitWithMessage("Could not open channel.");
            throw new NullReferenceException("Channel is null.");
        }
        return channel;
    }
    /**
    <summary>
        Method to check if a json token is null. If the jtoken isn't null then the token
        will be returned as a non-nullable reference. Otherwise an exception will be thrown.
    </summary>
    */
    [Obsolete] ///Replaced by more generic method
    private static JToken GetNonNullToken(JToken? j)
    {
        /*if(j == null){throw new NullReferenceException("Provided object is null.");}
        return (JToken)j; //Ignore warning as check has been done*/
        return GetNonNullValue<JToken>(j);
    }

    private static void StartConsumers(IModel channel, IConfigurationSection config, RabbitMQSettings rmq, DatabaseAccessor dbAccessor){
        //Create Messager
        Console.WriteLine("Creating QueueMessagerService...");
        var qms = new QueueMessagerService(rmq);
        Console.WriteLine("QueueMessagerService created.");
        //Raw Data consumer
        Console.WriteLine("Creating Raw Data Consumers...");
        //Get consumer list from config as json string
        var consumerListObject = GetJsonFromConfiguration(config.GetSection("ProcessingConsumers"));
        var consumerList = JsonObjectsToJArray(consumerListObject);
        //Convert json string to list of consumers
        var consumers = JsonConvert.DeserializeObject<List<ConsumerSettings>>(consumerList.ToString());
        foreach (var consumer in consumers ?? throw new NullReferenceException("Consumer list is null."))
        {
            //Create consumer
            var rawDataConsumer = new RawDataConsumer(channel, dbAccessor, qms, consumer);
            //Start consuming
            channel.BasicConsume(
                queue: consumer.InputQueueId,
                autoAck: false,
                consumer: rawDataConsumer
            );
        }
        Console.WriteLine("Raw Data Consumers created.");
        Console.WriteLine("Creating Processed Data Consumers...");
        //Get consumer list from config as json string
        consumerListObject = GetJsonFromConfiguration(config.GetSection("ThreatDetectionConsumers"));
        consumerList = JsonObjectsToJArray(consumerListObject);
        //Convert json string to list of consumers
        consumers = JsonConvert.DeserializeObject<List<ConsumerSettings>>(consumerList.ToString());
        foreach (var consumer in consumers ?? throw new NullReferenceException("Consumer list is null."))
        {
            //Create consumer
            var processedDataConsumer = new RawDataConsumer(channel, dbAccessor, qms, consumer);
            //Start consuming
            channel.BasicConsume(
                queue: consumer.InputQueueId,
                autoAck: false,
                consumer: processedDataConsumer
            );
        }
        Console.WriteLine("Processed Data Consumers created.");
    }

    private static DatabaseAccessor ConnectDB(IConfigurationSection dbSettings){
        Console.WriteLine("Connecting to database...");
        //Get db server name, db name, username and password from appsettings.json
        
        var databaseAccessor = GetDatabaseAccessor(
            GetNonNullValue(dbSettings["DatabaseServer"]).ToString(),
            GetNonNullValue(dbSettings["DatabaseName"]).ToString(),
            GetNonNullValue(dbSettings["DatabaseUsername"]).ToString(),
            GetNonNullValue(dbSettings["DatabasePassword"]).ToString()    
        );
        //Test if the connection is valid
        if (!databaseAccessor.TestConnection())
        {
            ExitWithMessage("Could not connect to database. Please check the connection string in Configuration");
        }
        Console.WriteLine("Connected to database.");
        return databaseAccessor;
    }
}