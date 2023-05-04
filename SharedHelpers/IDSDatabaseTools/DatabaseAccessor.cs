//This class will be used to connect to the IDS db.
using DataModels;
namespace IDSDatabaseTools;
public class DatabaseAccessor{
    private IdsDbContext _connectionContext;

    public DatabaseAccessor(string connectionString){
        _connectionContext = new IdsDbContext(connectionString);
    }

    public void AddRawData(RawData rawData){
        _connectionContext.Add(rawData);
        _connectionContext.SaveChanges();
    }

    public void AddProcessedData(ProcessedData processedData){
        _connectionContext.Add(processedData);
        _connectionContext.SaveChanges();
    }

    public void AddResultsData(ResultsData resultsData){
        _connectionContext.Add(resultsData);
        _connectionContext.SaveChanges();
    }
    
    public RawData? GetRawDataWithId(long id){
        return _connectionContext.RawData.Find(id);
    }

    public bool TestConnection()
    {
        return _connectionContext.Database.CanConnect();
    }
}