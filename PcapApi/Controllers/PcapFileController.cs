using Microsoft.AspNetCore.Mvc;
using IDSDatabaseTools;
using DataModels;
using SharedHelpers.QueueTools;

namespace PcapApi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PcapFileController : ControllerBase
    {
        private readonly DatabaseAccessor _databaseAccessor;
        private readonly QueueMessagerService _queueMessagerService;
        private readonly string successQueueId;
        private readonly string failureQueueId;

        public PcapFileController(DatabaseAccessor databaseAccessor, QueueMessagerService queueMessagerService, QueueMappings queueMappings)
        {
            successQueueId = queueMappings.GetQueueId("RawDataSuccessQueue");
            failureQueueId = queueMappings.GetQueueId("RawDataFailedQueue");
            _databaseAccessor = databaseAccessor;
            _queueMessagerService = queueMessagerService;
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
            try{
            long id = _databaseAccessor.AddRawData(new RawData(fileBytes));
            //send message to success queue
            //Get the raw data id from the database

            _queueMessagerService.SendMessage(id.ToString(),successQueueId);

            return Ok("File uploaded successfully");
            }
            catch(Exception e){
                //send message to failure queue
                _queueMessagerService.SendMessage(e.Message,failureQueueId);
                return BadRequest("File upload failed. Message sent to failure queue");
            }
        }
    }
}
