using Newtonsoft.Json;

public class ConsumerSettings
{
    [JsonProperty("ConsumerAmount")]
    public int ConsumerAmount { get; set; }
    [JsonProperty("ConsumerName")]
    public string ConsumerName { get; set; }
    [JsonProperty("InputQueueId")]
    public string InputQueueId { get; set; }
    [JsonProperty("OutputQueueId")]
    public string OutputQueueId { get; set; }
    [JsonProperty("FailedQueueId")]
    public string FailedQueueId { get; set; }
    [JsonProperty("Settings")]
    public ConsumerSettingsDetails Settings { get; set; }

    public ConsumerSettings()
    {
        Settings = new ConsumerSettingsDetails();
        ConsumerName = "";
        InputQueueId = "";
        OutputQueueId = "";
        FailedQueueId = "";
        ConsumerAmount = 0;
    }
    [JsonConstructor]
    public ConsumerSettings(string consumerName, string inputQueueId, string outputQueueId, string failedQueueId, ConsumerSettingsDetails settings, int consumerAmount = 1)
    {
        ConsumerName = consumerName;
        InputQueueId = inputQueueId;
        OutputQueueId = outputQueueId;
        FailedQueueId = failedQueueId;
        Settings = settings;
        ConsumerAmount = consumerAmount;
    }
}
public class ConsumerSettingsDetails
{
    [JsonProperty("BaseCommand")]
    public string BaseCommand { get; set; }
    [JsonProperty("ExecutablePath")]
    public string ExecutablePath { get; set; }
    [JsonProperty("ExecutableArguments")]
    public string ExecutableArguments { get; set; }
    [JsonProperty("OutputPath")]
    public string OutputPath { get; set; }

    public ConsumerSettingsDetails()
    {
        BaseCommand = "";
        ExecutablePath = "";
        ExecutableArguments = "";
        OutputPath = "";
    }
    [JsonConstructor]
    public ConsumerSettingsDetails(string baseCommand, string executablePath, string executableArguments, string outputPath)
    {
        BaseCommand = baseCommand;
        ExecutablePath = executablePath;
        ExecutableArguments = executableArguments;
        OutputPath = outputPath;
    }
}