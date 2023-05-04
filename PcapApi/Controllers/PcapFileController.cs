using Microsoft.AspNetCore.Mvc;
using static SharedHelpers.HelperFunctions;


namespace PcapApi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PcapFileController : ControllerBase
    {
        [HttpPost]
        public IActionResult UploadPcapFile(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("Invalid file");
            }

            //Connect to DB
            //Get project directory
            var projectDirectory = Directory.GetCurrentDirectory();
            //Build the configuration object
            var config = new ConfigurationBuilder()
            .AddJsonFile(projectDirectory + "\\appConfig.json", optional: false, reloadOnChange: true)
            .Build();
            //Get db server name, db name, username and password from appsettings.json
            var dbSettings = config.GetSection("DatabaseSettings");
            var databaseAccessor = GetDatabaseAccessor(
                GetNonNullValue(dbSettings["DatabaseServer"]).ToString(),
                GetNonNullValue(dbSettings["DatabaseName"]).ToString(),
                GetNonNullValue(dbSettings["DatabaseUsername"]).ToString(),
                GetNonNullValue(dbSettings["DatabasePassword"]).ToString()
            );
            //Test if the connection is valid
            if (!databaseAccessor.TestConnection())
            {
                return StatusCode(500, "Could not connect to database. Please check the connection string in appsettings.json");
            }
            return Ok("File uploaded and saved");
        }
    }
}
