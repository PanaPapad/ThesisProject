using System.Text;
using RabbitMQ.Client;

namespace SharedHelpers.QueueTools;
public class QueueMessagerService : IDisposable
{
    private readonly IConnection _connection;
    private readonly IModel _channel;
    private readonly Dictionary<string, string> _queueNames;

    public QueueMessagerService(RabbitMQSettings _rabbitMqSettings)
    {
        _connection = _rabbitMqSettings.GetConnectionFactory();
        _channel = _connection.CreateModel();
        //Declare queues
        foreach(var queue in _rabbitMqSettings.Queues){
            queue.Declare(_channel);
        }
        //Get queue names
        _queueNames = new Dictionary<string, string>();
        foreach(var queue in _rabbitMqSettings.Queues){
            _queueNames.Add(queue.GetQueueId(), queue.GetQueueName());
        }
    }

    public void SendMessage(string message, string queueId){
        if (!_queueNames.TryGetValue(queueId, out var queueName))
        {
            throw new ArgumentException($"Queue with ID '{queueId}' was not found.");
        }
        var body = Encoding.UTF8.GetBytes(message);
        _channel.BasicPublish(
            exchange: "",
            routingKey: _queueNames[queueId],
            basicProperties: null,
            body: body
        );
    }

    public void Dispose()
    {
        _channel.Close();
        _connection.Close();
    }
}