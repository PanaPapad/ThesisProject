using IDSDatabaseTools;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json.Linq;

namespace SharedHelpers;
/**
<summary>
    This static class contains generic helper methods for the IDS app.
</summary>
*/
public static class HelperFunctions
{
    /**
    <summary>
        This helper method helps with null checking. If the provided object is null then
        an exception will be thrown. Otherwise the object will be returned as a non-nullable
        reference.
    </summary>
    */
    public static T GetNonNullValue<T>(T? t) where T : class
    {
        if (t == null) { throw new NullReferenceException("Provided object is null."); }
        return t; //Ignore warning as check has been done
    }
    /**
    <summary>
        This method returns a DatabaseAccessor object using the provided parameters as a connection string.
    </summary>
    */
    public static DatabaseAccessor GetDatabaseAccessor(string DbServer, string DbName, string DbUsername, string DbPassword)
    {
        var connectionString = $"Server={DbServer};Database={DbName};uid={DbUsername};Password={DbPassword};";
        return new DatabaseAccessor(connectionString);
    }
    /**
    <summary>
        This method returns an IConfigurationRoot object read from a JSON file.
    </summary>
    */
    public static IConfigurationRoot GetConfigFromJsonFile(string path)
    {
        return new ConfigurationBuilder()
        .AddJsonFile(path, optional: false, reloadOnChange: true)
        .Build();
    }
    public static JObject GetJsonFromConfiguration(IConfiguration config)
    {
        var jObject = new JObject();

        foreach (var child in config.GetChildren())
        {
            if (child.Value == null)
            {
                // This is a sub-section, add it as a nested JObject
                jObject[child.Key] = GetJsonFromConfiguration(child);
            }
            else
            {
                // This is a value, add it directly
                jObject[child.Key] = new JValue(child.Value);
            }
        }

        return jObject;
    }
    public static JArray JsonObjectsToJArray(JObject jObject)
    {
        JArray jArray = new JArray(jObject.Properties().OrderBy(p => p.Name).Select(p => p.Value));
        return jArray;
    }
}