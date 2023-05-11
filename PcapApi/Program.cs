using IDSDatabaseTools;
using Microsoft.AspNetCore.Http.Features;
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
try
{
    builder.Services.AddScoped(x => new DatabaseAccessor(conn));
}
catch (Exception e)
{
    Console.WriteLine("Could not connect to database. Please check your configuration and try again.");
    Console.WriteLine(e.Message);
    //Wait for user input to exit
    Console.WriteLine("Press [enter] to exit.");
    Console.ReadLine();
    Environment.Exit(1);
}
builder.Services.AddSingleton<QueueMappings>
    (new QueueMappings(config.GetSection("ControllerQueueMappings")));
try
{
    builder.Services.AddSingleton<QueueMessagerService>
        (new QueueMessagerService(new RabbitMQSettings(queueSettings)));
}
catch (RabbitMQ.Client.Exceptions.BrokerUnreachableException e)
{
    Console.WriteLine("Could not connect to RabbitMQ. Please check your configuration and try again.");
    Console.WriteLine(e.Message);
    //Wait for user input to exit
    Console.WriteLine("Press [enter] to exit.");
    Console.ReadLine();
    Environment.Exit(1);
}
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
//Increase the file size limit to 10MB
builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 1024 * 1024 * 10;
});


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
