using Microsoft.Extensions.Configuration;

    /**
    <summary>
        This class is used to map queue ids to queue names. This is done to avoid hardcoding
        queue ids in the code. These mappings are typically stored in a config file in either
        JSON or XML format.
    </summary>
    */
public class QueueMappings{

    private Dictionary<string, string> _queueIds;
    public QueueMappings(IConfigurationSection queueMappings){
        _queueIds = new Dictionary<string, string>();
        foreach(var queue in queueMappings.GetChildren()){
            _queueIds.Add(queue.Key, queue.Value ??
                 throw new ArgumentException($"Failed to create queue mapping for {queue.Key}"));
        }
    }

    /**
    <summary>
        This method returns the queue id for the provided queue name.
    </summary>
    */
    public string GetQueueId(string queueMapping){
        if(!_queueIds.TryGetValue(queueMapping, out var queueId)){
            throw new KeyNotFoundException($"Queue mapping '{queueMapping}' was not found.");
        }
        return queueId;
    }
    
}