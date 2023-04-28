param(
    [string]$interface = "Wi-Fi",
    [int]$duration = 5, # in minutes
    [string]$pcap_dir = (Join-Path $PSScriptRoot "..\StoredPcapFiles")
)

# Check if Wireshark is installed
if (!(Get-Command tshark.exe -ErrorAction SilentlyContinue)) {
    Write-Output "Wireshark is not installed."
    exit 1
}
#convert duration to seconds
$duration = $duration * 60
# find the path to tshark.exe
$wireshark_dir = (Get-Command tshark.exe).Source
# Get the current timestamp
$timestamp = (Get-Date -Format "HH_mm_ss")
# Start the Wireshark capture
try{
    & "$wireshark_dir" -i $interface -a duration:$duration -w "$pcap_dir\capture_$timestamp.pcap"
}
catch {
    Write-Output "Failed to start Wireshark capture."
    Write-Output $_.Exception.Message
    exit 1
}
