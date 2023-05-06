using Microsoft.Extensions.Configuration;

public class QueueMappings{
    private Dictionary<string, string> _queueIds;
    public QueueMappings(IConfigurationSection queueMappings){
        _queueIds = new Dictionary<string, string>();
        foreach(var queue in queueMappings.GetChildren()){
            _queueIds.Add(queue.Key, queue.Value ??
                 throw new ArgumentException($"Failed to create queue mapping for {queue.Key}"));
        }
    }

    public string GetQueueId(string queueMapping){
        if(!_queueIds.TryGetValue(queueMapping, out var queueId)){
            throw new ArgumentException($"Queue mapping '{queueMapping}' was not found.");
        }
        return queueId;
    }
    
}