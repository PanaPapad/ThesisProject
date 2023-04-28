$global:requirements_met = $true
function CheckRequirement {
    param(
        [string]$Message,
        [scriptblock]$CheckScript,
        [string]$FailedMessage
    )
    # Display the message
    Write-Host $Message
    # Run the check script
    $result = Invoke-Command -ScriptBlock $CheckScript
    # If the check script returns false warn the user and set requirements_met to false
    if (-not $result) {
        Write-Warning $FailedMessage
        $global:requirements_met = $false
    }
    else{
        Write-Host -NoNewline "OK`n"
    }
}
# Check that PowerShell version is 5.0 or higher
CheckRequirement `
    -Message "Checking PowerShell version..." `
    -CheckScript { $PSVersionTable.PSVersion.Major -ge 5 } `
    -FailedMessage "This script requires PowerShell version 5.0 or higher."
  
# Check if Wireshark is installed
CheckRequirement `
    -Message "Checking Wireshark..." `
    -CheckScript { Get-Command "tshark.exe" -ErrorAction SilentlyContinue } `
    -FailedMessage "Wireshark is not installed."

# Check if CiCFlowmeter is installed
# Get the path from the config.xml
[xml]$config = Get-Content -Path "$PSScriptRoot\config.xml"
$cfm_path = $config.SelectSingleNode("//CiCFlowmeterSettings/flowmeterPath").InnerText
CheckRequirement `
    -Message "Checking CiCFlowmeter..." `
    -CheckScript { Test-Path $cfm_path } `
    -FailedMessage "CiCFlowmeter is not installed or the path is incorrect."
  
# Check that java is installed
CheckRequirement `
    -Message "Checking Java..." `
    -CheckScript { Get-Command "java.exe" -ErrorAction SilentlyContinue } `
    -FailedMessage "Java is not installed."
  
# Check that the JAVA_HOME environment variable is set
CheckRequirement `
    -Message "Checking JAVA_HOME environment variable..." `
    -CheckScript { [string]::IsNullOrEmpty($env:JAVA_HOME) -eq $false } `
    -FailedMessage "JAVA_HOME is not set."
  
# Check if user is an administrator
CheckRequirement `
    -Message "Checking if user is an administrator..." `
    -CheckScript {
    $user = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($user)
    $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
} `
    -FailedMessage "User is not an Administrator."
  
# Check if all requirements are met
if ($global:requirements_met) {
    Write-Host "All requirements are met."
    exit 0
}
else {
    Write-Host "Not all requirements are met."
    exit 1
}
  