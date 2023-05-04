using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static SharedHelpers.HelperFunctions;
namespace DataProcessor;
public class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Starting DataProcessor...");
        Console.WriteLine("Reading appsettings.json...");
        //Get project directory
        var projectDirectory = Directory.GetCurrentDirectory();
        //Build the configuration object
        var config = new ConfigurationBuilder()
        .AddJsonFile(projectDirectory + "\\appConfig.json", optional: false, reloadOnChange: true)
        .Build();
        Console.WriteLine("appsettings.json read.");

        Console.WriteLine("Connecting to database...");
        //Get db server name, db name, username and password from appsettings.json
        var dbSettings = config.GetSection("DatabaseSettings");
        var databaseAccessor = GetDatabaseAccessor(
            GetNonNullValue(dbSettings["DatabaseServer"]).ToString(),
            GetNonNullValue(dbSettings["DatabaseName"]).ToString(),
            GetNonNullValue(dbSettings["DatabaseUsername"]).ToString(),
            GetNonNullValue(dbSettings["DatabasePassword"]).ToString()    
        );
        //Test if the connection is valid
        if (!databaseAccessor.TestConnection())
        {
            ExitWithMessage("Could not connect to database. Please check the connection string in appsettings.json");
        }
        Console.WriteLine("Connected to database.");

        Console.WriteLine("Connecting to RabbitMQ...");
        //Connect to RabbitMQ
        var RabbitMQSettings = config.GetSection("RabbitMQSettings");
        var factory = new ConnectionFactory()
        {
            HostName = RabbitMQSettings["RabbitMQHost"],
            UserName = RabbitMQSettings["RabbitMQUsername"],
            Password = RabbitMQSettings["RabbitMQPassword"],
        };
        using (var connection = factory.CreateConnection())
        using (var channel = connection.CreateModel())
        {
            Console.WriteLine("Connected to RabbitMQ.");
            Console.WriteLine("Declaring queues...");
            DeclareQueues(RabbitMQSettings, channel);
            Console.WriteLine("Queues declared.");
        }

    }

    
    ///<summary>
    ///Helper method to exit the program with a message.
    ///</summary>
    private static void ExitWithMessage(string message)
    {
        Console.WriteLine(message);
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
    
}