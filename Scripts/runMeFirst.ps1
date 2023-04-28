#Helper functions

#Function to call other scripts
function RunScript {
    param(
        [string]$scriptName,
        [string[]]$argns
    )

    try {
        Write-Host "--Running script '$scriptName'"
        $psi = [PowerShell]::Create()
        $psi.AddScript("&'$PSScriptRoot\\$scriptName' $argns") | Out-Null
        $psi.Invoke() | Out-Null
        $psi.Streams.Error | ForEach-Object { Write-Error $_ }
        $psi.Streams.Warning | ForEach-Object { Write-Warning $_ }
        $psi.Streams.Verbose | ForEach-Object { Write-Verbose $_ }
        $psi.Streams.Debug | ForEach-Object { Write-Debug $_ }
        $psi.Streams.Information | ForEach-Object { Write-Host $_.MessageData }

        $exitCode = $psi.Runspace.SessionStateProxy.GetVariable("LASTEXITCODE")
        Write-Host "--Script '$scriptName' finished."
        return $exitCode
    }
    catch {
        Write-Host "--Failed to run script '$scriptName'."
        Write-Host $_.Exception.Message
        Exit 1
    }
    finally {
        $psi.Dispose()
    }
}

#Exit if not running as admin
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) { Write-Warning "You do not have Administrator rights to run this script!`nPlease re-run this script as an Administrator!" ; break }

# run the CheckRequirements.ps1 script to check if environment is ready
Write-Host "Checking environment..."
$exitCode = RunScript "CheckRequirements.ps1"
#if exit code is not 0 then exit
if ($exitCode -ne 0) {
    Write-Warning "Environment is not ready."
    Exit 1
}
Write-Host "Environment is ready.`n"
# call the setupTaskScheduler.ps1 script to create the capture task
Write-Host "Setting up task scheduler..."
$taskSetup = RunScript "setupTaskScheduler.ps1"
if($taskSetup -ne 0) {
    Write-Warning "Failed to setup task scheduler."
    Exit 1
}
Write-Host "Task scheduler setup successfully.`n"
Write-Host "Setup complete."
Pause
Exit 0



