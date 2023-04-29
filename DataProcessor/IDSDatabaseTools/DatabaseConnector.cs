//This class will be used to connect to the IDS db.
using Microsoft.EntityFrameworkCore;

public class DatabaseAccessor{
    private DbContext _connectionContext;

    public DatabaseAccessor(string connectionString){
        _connectionContext = new IdsDbContext(connectionString);
    }
}