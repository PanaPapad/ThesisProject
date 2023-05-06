using Newtonsoft.Json;
using Microsoft.Extensions.Configuration;
using RabbitMQ.Client;

namespace SharedHelpers.QueueTools;

public class RabbitMQSettings
{
    public string HostName { get; set; }
    public int Port { get; set; }
    public string UserName { get; set; }
    public string Password { get; set; }
    public List<RabbitMQueue> Queues { get; set; }

    public RabbitMQSettings(string hostName, int port = 5672, string userName = "", string password = "")
    {
        HostName = hostName;
        Port = port;
        UserName = userName;
        Password = password;
        Queues = new List<RabbitMQueue>();
    }

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

    public void AddQueuesFromJson(string json)
    {
        var queues = JsonConvert.DeserializeObject<List<RabbitMQueue>>(json);
        //Check if queues is null
        if(queues != null){
            Queues.AddRange(queues);
        }
    }

    public void AddQueue(string queueId, string queueName, bool isDurable = false, bool isExclusive = false, bool isAutoDelete = false)
    {
        Queues.Add(new RabbitMQueue(queueId, queueName, isDurable, isExclusive, isAutoDelete));
    }

    public void DeclareQueues(IModel channel)
    {
        foreach (var queue in Queues)
        {
            queue.Declare(channel);
        }
    }

}