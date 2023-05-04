using Microsoft.AspNetCore.Mvc;
using IDSDatabaseTools;


namespace PcapApi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PcapFileController : ControllerBase
    {
        private readonly DatabaseAccessor _databaseAccessor;

        public PcapFileController(DatabaseAccessor databaseAccessor)
        {
            _databaseAccessor = databaseAccessor;
        }
        [HttpPost]
        public IActionResult UploadPcapFile(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("Invalid file");
            }
            if(!_databaseAccessor.TestConnection()){
                return BadRequest("Could not connect to DB");}           
            return Ok("Connected to DB");
        }
    }
}
