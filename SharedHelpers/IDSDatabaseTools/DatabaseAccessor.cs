//This class will be used to connect to the IDS db.
using DataModels;
namespace IDSDatabaseTools;
public class DatabaseAccessor{
    private IdsDbContext _connectionContext;

    public DatabaseAccessor(string connectionString){
        _connectionContext = new IdsDbContext(connectionString);
    }

    public long AddRawData(RawData rawData){
        _connectionContext.RawData.Add(rawData);
        _connectionContext.SaveChanges();
        return rawData.Id;
    }

    public long AddProcessedData(ProcessedData processedData){
        _connectionContext.Add(processedData);
        _connectionContext.SaveChanges();
        return processedData.Id;
    }

    public long AddResultsData(ResultsData resultsData){
        _connectionContext.Add(resultsData);
        _connectionContext.SaveChanges();
        return resultsData.Id;
    }
    
    public RawData? GetRawDataWithId(long id){
        return _connectionContext.RawData.Find(id);
    }

    public bool TestConnection()
    {
        return _connectionContext.Database.CanConnect();
    }
}