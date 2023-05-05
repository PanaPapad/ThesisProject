using Microsoft.AspNetCore.Mvc;
using IDSDatabaseTools;
using DataModels;

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
        public async Task<IActionResult> UploadPcapFile(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("Invalid file");
            }
            if (!_databaseAccessor.TestConnection())
            {
                return BadRequest("Could not connect to DB");
            }
            //Use database accessor to save file in a new raw data record
            byte[] fileBytes;
            using (var memoryStream = new MemoryStream())
            {
                await file.CopyToAsync(memoryStream);
                fileBytes = memoryStream.ToArray();
            }
            _databaseAccessor.AddRawData(new RawData(fileBytes));
            return Ok("File uploaded successfully");
        }
    }
}
