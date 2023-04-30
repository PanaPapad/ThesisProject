using Microsoft.Extensions.Configuration;
public class Program
{
    private static void Main(string[] args)
    {
        //Build the configuration object
        var config = new ConfigurationBuilder()
        .AddJsonFile("appConfig.json", optional: false, reloadOnChange: true)
        .Build();

        //Get db server name, db name, username and password from appsettings.json
        var dbServerName = config["DatabaseServer"];
        var dbName = config["DatabaseName"];
        var dbUsername = config["DatabaseUsername"];
        var dbPassword = config["DatabasePassword"];
        //build the connection string
        var connectionString = $"Server={dbServerName};Database={dbName};uid={dbUsername};Password={dbPassword};";
        //create a new database accessor
        var databaseAccessor = new DatabaseAccessor(connectionString);
        //get the first raw data object from the database
        var rawData = databaseAccessor.GetRawDataWithId(1);
        //print the raw data object to the console
        Console.WriteLine(rawData.Id);
    }
}