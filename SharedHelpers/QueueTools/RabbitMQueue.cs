using RabbitMQ.Client;
using Newtonsoft.Json;
public class RabbitMQueue{
    [JsonProperty("id")]
    private string _queueId;
    [JsonProperty("name")]
    private string _queueName;
    [JsonProperty("durable")]
    private bool _isDurable;
    [JsonProperty("exclusive")]
    private bool _isExclusive;
    [JsonProperty("autoDelete")]
    private bool _isAutoDelete;
    [JsonProperty("arguments")]
    private Dictionary<string, object> _arguments;

    [JsonConstructor]
    public RabbitMQueue(string id, string name, bool durable = false, bool autoDelete = false, bool exclusive = false){
        _queueId = id;
        _queueName = name;
        _isDurable = durable;
        _isExclusive = exclusive;
        _isAutoDelete = autoDelete;
        _arguments = new Dictionary<string, object>();
    }

    public string GetQueueId(){
        return _queueId;
    }

    public string GetQueueName(){
        return _queueName;
    }

    public void Declare(IModel channel){
        channel.QueueDeclare(queue: _queueName,
                             durable: _isDurable,
                             exclusive: _isExclusive,
                             autoDelete: _isAutoDelete,
                             arguments: _arguments
                             );
    }
}