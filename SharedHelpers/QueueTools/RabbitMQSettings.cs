using Newtonsoft.Json;
using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;

namespace SharedHelpers.QueueTools;

/**
<summary>
    This class is used to store the settings for a RabbitMQ connection.
    Methods to make use of the settings are also provided.
    Queue definitions can be added from a json file or manually.
</summary>
*/
public class RabbitMQSettings
{
    public string HostName { get; set; }
    public int Port { get; set; }
    public string UserName { get; set; }
    public string Password { get; set; }
    public List<RabbitMQueue> Queues { get; set; }

    /**
    <summary>
        Creates a new instance of RabbitMQSettings.
    </summary>
    */
    public RabbitMQSettings(string hostName, int port = 5672, string userName = "", string password = "")
    {
        HostName = hostName;
        Port = port;
        UserName = userName;
        Password = password;
        Queues = new List<RabbitMQueue>();
    }

    /**
    <summary>
        Creates a new instance of RabbitMQSettings from an IConfigurationSection.
        Properties are read from the section where keys are expected to be named as follows:
        RabbitMQ[PropertyName] e.g. RabbitMQHost
    </summary>
    */
    public RabbitMQSettings(IConfigurationSection config)
    {
        HostName = config["RabbitMQHost"] ?? "";
        Port = int.Parse(config["RabbitMQPort"] ?? "5672");
        UserName = config["RabbitMQUsername"] ?? "";
        Password = config["RabbitMQPassword"] ?? "";
        string? queuesJsonFile = config["RabbitMQDefinitionsFilePath"];
        Queues = new List<RabbitMQueue>();
        if(queuesJsonFile != null){
            //Open json file
            string queuesJson = File.ReadAllText(queuesJsonFile);
            AddQueuesFromJson(queuesJson);
        }
    }
    /**
        <summary>
        Create a connection factory from the settings.
        </summary>
    */
    public IConnection GetConnectionFactory()
    {
        var factory = new ConnectionFactory()
        {
            HostName = HostName,
            Port = Port,
            UserName = UserName,
            Password = Password
        };
        return factory.CreateConnection();
    }

    /**
    <summary>
        Adds queues from a json string.
        The json string should be an array of RabbitMQueue objects.
    </summary>
    */
    public void AddQueuesFromJson(string json)
    {
        var queues = JsonConvert.DeserializeObject<List<RabbitMQueue>>(json);
        //Check if queues is null
        if(queues != null){
            Queues.AddRange(queues);
        }
    }

    /**
        <summary>
        Adds a RabbitMQueue to the list of queues.
        </summary>
    */
    public void AddQueue(string queueId, string queueName, bool isDurable = false, bool isExclusive = false, bool isAutoDelete = false)
    {
        Queues.Add(new RabbitMQueue(queueId, queueName, isDurable, isExclusive, isAutoDelete));
    }

    /**
        <summary>
        Delcare all queues in the list of queues on the provided channel. 
        </summary>
    */
    public void DeclareQueues(IModel channel)
    {
        foreach (var queue in Queues)
        {
            queue.Declare(channel);
        }
    }

}