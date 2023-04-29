//This is the main class of the DataProcessor .Net core application.
//When a RabbitMQ messge is received, an asynchronous task is started to process the message.
//Tasks make use of workers to process the message.
//Any errors are logged to the console.
//The application is configured using the appConfig.xml file.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Xml.Linq;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;

class DataProcessor{

    static void Main(string[] args){
        //read the configuration file
        XDocument config = XDocument.Load("appConfig.xml");
        
    }
}
