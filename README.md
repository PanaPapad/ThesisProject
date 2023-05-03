# ThesisProject
Repo for my CS degree thesis project

# Description
This project is a collection of components that are used to monitor a BACnet network.
This project is directed towards network administrators that want to monitor their BACnet network for possible attacks or malfunctions.

The project is composed of 5 main components:
- A data collection script that utilizes tshark to capture network traffic from a BACnet IP network.
- A Database that stores the data in all of its stages.
- CiCFlowMeter, a tool that extracts features from the network traffic. This tool was developed the following team of researchers:
    Arash Habibi Lashkari: Researcher and Developer (Founder)
    Gerard Drapper: Researcher and Developer (Co-funder)
    muhammad saiful islam: Researcher and Developer (Co-funder)
    for more information about the tool, please visit: https://github.com/ahlashkari/CICFlowMeter
- A data processing module.
- An IDS that uses ML algorithms to detect anomalies in the network traffic.
- A web application that displays the data collected by the IDS.

# Data collection script
This script is written in Powershell and it uses tshark to capture network traffic from a BACnet IP network.
The script will capture traffic for a given interval on a given interface.
The traffic is saved in a .pcap file in a directory of your choosing.
Each .pcap file has a name appended with the date and time of the capture.

In order to automate the data collection process, the script can be scheduled to run at a given interval using the Windows Task Scheduler. A script that automates the scheduling process is provided in the repository.

# CiCFlowMeter
This tool is used to extract features from the network traffic.
For more information about the tool, please visit: https://github.com/ahlashkari/CICFlowMeter

# Data processing module
This application consists of 2 main components:
- A queuing system for the data in all of its stages.
- The data processing module itself. Which is written in C#.

    # Queuing system
    The queing system used is RabbitMQ. It is used to queue the data in all of its stages in seperate queues.
    Data stages are as follows:
    - Raw data. -- .pcap files.
    - Processed data. -- .csv files from CiCFlowmeter.
    - Results from the IDS. -- .csv? files.
    There are 2 queues for each stage:
    - A queue for the raw data.
    - A queue for any data that failed to be processed in the current stage.

    # Data processing module
    This module is written in C# and it is responsible for processing the data in all of its stages.
    Each step in the data processing pipeline is a worker that is responsible for processing the data in its stage.
    Workers begin working when they receive a message from the relevant queue.
    Workers run in parallel.

# IDS
This IDS is written in Python and it uses ML algorithms to detect anomalies in the network traffic.
It was developed by researchers at the Network Lab of the University of Cyprus.
!REMEMBER TO ADD CITATION!

# Web application
This web app is written in php using the Laravel framework.
It is used to display the results of the IDS.
The web app was developed by researchers at the Network Lab of the University of Cyprus.
!REMEMBER TO ADD CITATION!

# Database
The database used is MySQL.
The database is used to store the data in all of its stages.
The database is also used to store the results of the IDS.
The data is stored as Blobs in the database.
Each record has a timestamp of entry and timestamp of collection.
Processed data is assosciated with the raw data that it was extracted from.
Results from the IDS are assosciated with the processed data that they were extracted from.
The script used to create the database is provided in the repository.

# Requirements
    # Data collection script
    - Windows 10 or later.
    - Powershell 5.1 or later.
    - Wireshark.

    # CiCFlowMeter
    - Java 8 or later.

    # Data processing module
    - .NET Core Runtime 3.1 or later.
    - Erlang 25.3
    - RabbitMQ Server 3.11.13.

    # IDS
    - Python 3.7 or later.

    # Web application
    - PHP
    - Web server (Apache, Nginx, etc.)

    # Database
    - MariaDB 10.4 or later.