<#
Push the local core/ to a fresh AutoDL instance, run setup, and a 2-minute smoke train.
Usage (from the repo root on your Windows box):
  powershell scripts/deploy_ssh.ps1 -HostAddress <autodl-host> -Port <ssh-port> [-Key <path-to-key>]
Example:
  powershell scripts/deploy_ssh.ps1 -HostAddress connect.cqa1.seetacloud.com -Port 12345
#>
param(
    [Parameter(Mandatory = $true)][string]$HostAddress,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$Key = "$env:USERPROFILE\.ssh\id_rsa",
    [string]$WorkDir = "/root/autodl-tmp/bit-jev"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host "== push code to ${HostAddress}:${Port} -> $WorkDir"
ssh -p $Port -i $Key -o StrictHostKeyChecking=accept-new root@$HostAddress "mkdir -p $WorkDir && rm -rf $WorkDir/core"
tar -czf - -C $repo core | ssh -p $Port -i $Key root@$HostAddress "cd $WorkDir && tar -xzf -"

Write-Host "== setup env (pip install)"
ssh -p $Port -i $Key root@$HostAddress "cd $WorkDir/core && bash scripts/setup_autodl.sh $WorkDir"

Write-Host "== smoke train (2 minutes)"
ssh -p $Port -i $Key root@$HostAddress "cd $WorkDir/core && bash scripts/smoke_train.sh"

Write-Host @"

done. next, on the instance:
  ssh -p $Port -i $Key root@$HostAddress
  cd $WorkDir/core && bash scripts/run_train.sh        # the release recipe (~1-2 h)
  cd $WorkDir/core && bash scripts/run_eval.sh          # benchmark against frozen suites
  cd $WorkDir/core && bash scripts/run_convert.sh       # export merged/ + head

and back here:
  powershell scripts/pull_results.ps1 -HostAddress $HostAddress -Port $Port
"@
