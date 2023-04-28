#Make sure to run as admin or exit
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "You do not have Administrator rights to run this script!`nPlease re-run this script as an Administrator!"
    Exit 1
}
try {
    # Load the XML settings file
    $doc = New-Object System.Xml.XmlDocument
    $doc.Load("./config.xml")

    # Get the CaptureSettings node
    $captureSettings = $doc.SelectSingleNode("//CaptureSettings")

    # Get the settings
    $interval = [int]$captureSettings.SelectSingleNode("captureInterval").InnerText
    $interface = $captureSettings.SelectSingleNode("captureInterface").InnerText
    $output_dir = $captureSettings.SelectSingleNode("captureOutDir").InnerText
    $taskname = $captureSettings.SelectSingleNode("captureTaskName").InnerText
    $startTime = Get-Date
    $endTime = $startTime.AddYears(50) # Set the end time to 50 years from start time
    $intervalTime = New-TimeSpan -Minutes $interval

    # Check if task exists and delete it if it does
    $task = Get-ScheduledTask -TaskName $taskname -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "Task already exists. Deleting it..."
        Unregister-ScheduledTask -TaskName $taskname -Confirm:$false
    }

    $powershellPath = (Get-Command powershell.exe).Source
    # Create the task to run the capture script with admin privileges
    $action = New-ScheduledTaskAction -Execute $powershellPath -Argument "-ExecutionPolicy Bypass -Command `"$PSScriptRoot\wiresharkCapture.ps1`" -interface `"$interface`" -duration $interval -pcap_dir `"$output_dir`""
    $trigger = New-ScheduledTaskTrigger -Once -At $startTime -RepetitionInterval $intervalTime -RepetitionDuration ($endTime - $startTime)
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
    $principal = New-ScheduledTaskPrincipal -UserID "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    Register-ScheduledTask -TaskName $taskname -Action $action -Trigger $trigger -Settings $settings -Principal $principal

    # Run on battery and ac power and allow multiple instances of the task to run
    $task = Get-ScheduledTask -TaskName $taskname
    $settings = $task.Settings
    $settings.MultipleInstances = "Parallel"
    $settings.StopIfGoingOnBatteries = $false
    $settings.DisallowStartIfOnBatteries = $false
    #task should not run longer than the interval plus 1 minute
    $settings.ExecutionTimeLimit = "PT" + ($interval + 1) + "M"
    #if task fails, restart every 1 minutes up to 10 times
    $settings.RestartCount = 10
    $settings.RestartInterval = "PT1M"
    Set-ScheduledTask -TaskName $taskname -Settings $settings

    # Trigger the task
    Start-ScheduledTask -TaskName $taskname
    Write-Host "Task created and triggered successfully."
    Exit 0
}
catch {
    Write-Warning "Failed to create task."
    Write-Warning $_.Exception.Message
    Exit 1
}
