<#
.SYNOPSIS
    Starts Project SETU development environment (Backend and Frontend) in two separate windows.

.HOW TO RUN
    From the repository root in PowerShell:
        .\scripts\start_dev.ps1

    To run in Demo Mode without requiring Meta or cloud API keys:
        $env:SETU_DEMO_MODE = "true"
        .\scripts\start_dev.ps1

    Or using the -Demo switch:
        .\scripts\start_dev.ps1 -Demo

.NOTE
    SETU_DEMO_MODE=true runs the system entirely offline without external API keys,
    returning validated benchmark fixtures and mock vibration telemetry for rapid local review.
#>

param(
    [switch]$Demo
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ($Demo -or ($env:SETU_DEMO_MODE -and $env:SETU_DEMO_MODE.ToLower() -in @("true", "1", "yes"))) {
    $env:SETU_DEMO_MODE = "true"
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " [SETU] DEMO MODE ENABLED (SETU_DEMO_MODE=true)" -ForegroundColor Cyan
    Write-Host " Running offline without requiring Meta / cloud API keys." -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
} else {
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host " [SETU] STARTING DEVELOPMENT SERVICES" -ForegroundColor Green
    Write-Host " (Tip: set `$env:SETU_DEMO_MODE='true' to run without API keys)" -ForegroundColor DarkGray
    Write-Host "==========================================================" -ForegroundColor Green
}

# 1. Start Backend in separate window
$demoEnvExport = if ($env:SETU_DEMO_MODE -eq "true") { "`$env:SETU_DEMO_MODE='true'; " } else { "" }
$backendCommand = "Set-Location '$RepoRoot'; $demoEnvExport Write-Host '[SETU Backend] Running on http://127.0.0.1:8000' -ForegroundColor Green; python -m uvicorn setu.api:app --port 8000"

Write-Host "-> Launching Backend window (uvicorn on port 8000)..." -ForegroundColor Yellow
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $backendCommand

# 2. Start Frontend in separate window
$frontendCommand = "Set-Location '$RepoRoot'; Write-Host '[SETU Frontend] Starting Vite dev server...' -ForegroundColor Green; npm run dev --prefix frontend"

Write-Host "-> Launching Frontend window (npm run dev --prefix frontend)..." -ForegroundColor Yellow
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $frontendCommand

Write-Host "`nAll development services launched in separate windows." -ForegroundColor Green
