using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Proxies;

public class IdsDbContext : DbContext{
    private readonly string _connectionString;
    public DbSet<RawData> RawData { get; set; }
    public DbSet<ProcessedData> ProcessedData { get; set; }
    public DbSet<ResultsData> ResultsData { get; set; }
    public IdsDbContext(string connectionString){
        _connectionString = connectionString;
        RawData = Set<RawData>();
        ProcessedData = Set<ProcessedData>();
        ResultsData = Set<ResultsData>();
    }
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder){
        optionsBuilder
            .UseLazyLoadingProxies()
            .UseMySql(_connectionString, new MySqlServerVersion(new Version(10,4,27)), options => options.EnableRetryOnFailure());
    }
    protected override void OnModelCreating(ModelBuilder modelBuilder){
        //Declare the primary keys
        modelBuilder.Entity<RawData>().HasKey(r => r.Id);
        modelBuilder.Entity<ProcessedData>().HasKey(p => p.Id);
        modelBuilder.Entity<ResultsData>().HasKey(r => r.Id);
        //Declare the relationships
        modelBuilder.Entity<ProcessedData>()
            .HasOne(p => p.RawData)
            .WithMany(r => r.ProcessedData)
            .HasForeignKey(p => p.RawDataId);
        modelBuilder.Entity<ResultsData>()
            .HasOne(r => r.ProcessedData)
            .WithMany(p => p.ResultsData)
            .HasForeignKey(r => r.ProcessedDataId);
    }
}