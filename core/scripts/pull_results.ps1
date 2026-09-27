<#
Pull trained runs back from AutoDL to local runs/.
Usage from the repo root:
  powershell scripts/pull_results.ps1 -HostAddress <host> -Port <port> [-Key <path>]
#>
param(
    [Parameter(Mandatory = $true)][string]$HostAddress,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$Key = "$env:USERPROFILE\.ssh\id_rsa",
    [string]$WorkDir = "/root/autodl-tmp/bit-jev",
    [string]$Out = "runs"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
scp -P $Port -i $Key -r root@${HostAddress}:$WorkDir/core/runs $Out
Write-Host "pulled into ./$Out"
