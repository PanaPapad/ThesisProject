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
    [JsonIgnore]
    private bool isDecalred = false;

    public RabbitMQueue(string queueId, string queueName, bool isDurable = false, bool isExclusive = false, bool isAutoDelete = false){
        _queueId = queueId;
        _queueName = queueName;
        _isDurable = isDurable;
        _isExclusive = isExclusive;
        _isAutoDelete = isAutoDelete;
    }

    public string GetQueueId(){
        return _queueId;
    }

    public string GetQueueName(){
        return _queueName;
    }

    public void SetDeclared(){
        isDecalred = true;
    }

    public void Declare(IModel channel){
        if(!isDecalred){
            channel.QueueDeclare(
                queue: _queueId,
                durable: _isDurable,
                exclusive: _isExclusive,
                autoDelete: _isAutoDelete,
                arguments: null
            );
            isDecalred = true;
        }
    }
}