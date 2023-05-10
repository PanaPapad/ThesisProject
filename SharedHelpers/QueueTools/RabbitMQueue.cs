using RabbitMQ.Client;
using Newtonsoft.Json;

/**
    <summary>
        This class represents a RabbitMQ queue. It can be used to declare a queue on a given channel.
        Objects from this class can be deserialized from a json string using the Newtonsoft Json package.
    </summary>
*/
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

    public void Declare(IModel channel, bool Override=false){
        //Check if queue exists
        bool queueExists = true;
        try{
            channel.QueueDeclarePassive(_queueName);
            //If no override return
            if(!Override){return;}
        }
        catch(RabbitMQ.Client.Exceptions.OperationInterruptedException){
            queueExists = false;
        }
        //Delete queue if it exists AND override is true
        if(queueExists){
            channel.QueueDelete(_queueName);
        }
        channel.QueueDeclare(queue: _queueName,
                             durable: _isDurable,
                             exclusive: _isExclusive,
                             autoDelete: _isAutoDelete,
                             arguments: _arguments
                             );
    }
}