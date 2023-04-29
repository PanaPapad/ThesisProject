using Microsoft.EntityFrameworkCore;

public class IdsDbContext : DbContext{
    private readonly string _connectionString;
    public DbSet<RawData> RawData { get; set; }
    public IdsDbContext(string connectionString){
        _connectionString = connectionString;
        RawData = Set<RawData>();
    }
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder){
        optionsBuilder.UseMySql(ServerVersion.AutoDetect(_connectionString), options => options.EnableRetryOnFailure());
    }
}