using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class Program
{
    private static void Main(string[] args)
    {   
        Console.WriteLine("Starting DataProcessor...");
        Console.WriteLine("Reading appsettings.json...");
        //Get project directory. Handle null warnings
        var projectDirectory = Directory.GetCurrentDirectory();
        //Build the configuration object
        var config = new ConfigurationBuilder()
        .AddJsonFile(projectDirectory + "\\appConfig.json", optional: false, reloadOnChange: true)
        .Build();
        Console.WriteLine("appsettings.json read.");
        Console.WriteLine("Connecting to database...");
        //Get db server name, db name, username and password from appsettings.json
        var dbSettings = config.GetSection("DatabaseSettings");
        var dbServerName = dbSettings["DatabaseServer"];
        var dbName = dbSettings["DatabaseName"];
        var dbUsername = dbSettings["DatabaseUsername"];
        var dbPassword = dbSettings["DatabasePassword"];
        //build the connection string
        var connectionString = $"Server={dbServerName};Database={dbName};uid={dbUsername};Password={dbPassword};";
        //create a new database accessor
        var databaseAccessor = new DatabaseAccessor(connectionString);
        //Test if the connection is valid
        if (!databaseAccessor.TestConnection())
        {
            ExitWithMessage("Could not connect to database. Please check the connection string in appsettings.json");
        }
        Console.WriteLine("Connected to database.");
        Console.WriteLine("Connecting to RabbitMQ...");
        //Connect to RabbitMQ
        var RabbitMQSettings = config.GetSection("RabbitMQSettings");
        var factory = new ConnectionFactory() { HostName = RabbitMQSettings["RabbitMQHost"],
                                                UserName = RabbitMQSettings["RabbitMQUsername"], 
                                                Password = RabbitMQSettings["RabbitMQPassword"],
                                                };
        using (var connection = factory.CreateConnection())
        using (var channel = connection.CreateModel())
        {
            Console.WriteLine("Connected to RabbitMQ.");
            Console.WriteLine("Declaring queues...");
            DeclareQueues(RabbitMQSettings, channel);
        }
            
    }

    private static void ExitWithMessage(string message)
    {
        Console.WriteLine(message);
        Environment.Exit(1);
    }

    private static void DeclareQueues(IConfigurationSection RabbitMQSettings,IModel channel){ 
        //Get the path to the json file containing the queue definitions
        string filePath = RabbitMQSettings["RabbitMQDefinitionsFilePath"] ?? "";
            if (filePath == "")
            {
                ExitWithMessage("RabbitMQDefinitionsFilePath not found in appsettings.json");
            }
            string jsonString = File.ReadAllText(filePath);
            JObject jsonData = GetNonNullValue<JObject>(JsonConvert.DeserializeObject(jsonString) as JObject);
            JArray? queuesList = GetNonNullToken(jsonData["queueList"]) as JArray;
            foreach (JObject queue in queuesList ?? new JArray())
            {
                try{
                channel.QueueDeclare(
                    GetNonNullToken(queue["name"]).ToString(),
                    GetNonNullToken(queue["durable"]).ToObject<bool>(),
                    GetNonNullToken(queue["exclusive"]).ToObject<bool>(),
                    GetNonNullToken(queue["autoDelete"]).ToObject<bool>(),
                    GetNonNullToken(queue["arguments"]).ToObject<IDictionary<string, object>>());
                }
                catch(NullReferenceException e){
                    Console.WriteLine(e.StackTrace);
                    ExitWithMessage("Queue definition is invalid or incomplete. Plese check the queueDefinitions.json.");
                }
            }     
        //Declare all queues in queuesList
        Console.WriteLine("Queues declared.");
    }

    //Method to check if a json token is null. If the jtoken isn't null then the token
    //will be returned as a non-nullable reference. Otherwise an exception will be thrown.
    private static JToken GetNonNullToken(JToken? j){
        /*if(j == null){throw new NullReferenceException("Provided object is null.");}
        return (JToken)j; //Ignore warning as check has been done*/
        return GetNonNullValue<JToken>(j);
    }
    private static T GetNonNullValue<T>(T? t) where T : class{
        if(t == null){throw new NullReferenceException("Provided object is null.");}
        return t; //Ignore warning as check has been done
    }
}