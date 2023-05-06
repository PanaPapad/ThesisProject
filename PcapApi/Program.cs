using IDSDatabaseTools;
using SharedHelpers;
using SharedHelpers.QueueTools;

var builder = WebApplication.CreateBuilder(args);

var projectDirectory = Directory.GetCurrentDirectory();
var config = new ConfigurationBuilder()
.AddJsonFile(projectDirectory + "\\appConfig.json", optional: false, reloadOnChange: true)
.Build();
var dbSettings = config.GetSection("DatabaseSettings");
string conn = HelperFunctions.GetNonNullValue(dbSettings["ConnectionString"]);
var queueSettings = config.GetSection("RabbitMQSettings");
builder.Services.AddScoped(x => new DatabaseAccessor(conn));
builder.Services.AddSingleton<QueueMappings>
    (new QueueMappings(config.GetSection("ControllerQueueMappings")));
builder.Services.AddSingleton<QueueMessagerService>
    (new QueueMessagerService(new RabbitMQSettings(queueSettings)));
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
