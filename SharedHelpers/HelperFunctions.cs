using IDSDatabaseTools;

namespace SharedHelpers;
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
    public static DatabaseAccessor GetDatabaseAccessor(string DbServer, string DbName, string DbUsername, string DbPassword)
    {
        var connectionString = $"Server={DbServer};Database={DbName};uid={DbUsername};Password={DbPassword};";
        return new DatabaseAccessor(connectionString);
    }
}