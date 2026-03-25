param(
  [string]$ListenAddress = "0.0.0.0",
  [int]$ListenPort = 50051,
  [string]$WslIp,
  [int]$WslPort = 50051,
  [string]$AllowedSubnet = "192.168.0.0/16"
)

if (-not $WslIp) {
  Write-Error "Pass -WslIp explicitly or resolve it from WSL before running this script."
  exit 1
}

$listenHost = "$ListenAddress`:$ListenPort"
$connectHost = "$WslIp`:$WslPort"

Write-Host "Configuring portproxy $listenHost -> $connectHost"
netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$ListenPort | Out-Null
netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$ListenPort connectaddress=$WslIp connectport=$WslPort

$ruleName = "gpu-dispatch-$ListenPort"
Write-Host "Refreshing firewall rule $ruleName for subnet $AllowedSubnet"
netsh advfirewall firewall delete rule name=$ruleName | Out-Null
netsh advfirewall firewall add rule name=$ruleName dir=in action=allow protocol=TCP localport=$ListenPort remoteip=$AllowedSubnet

Write-Host "Done. Validate with: netsh interface portproxy show v4tov4"
