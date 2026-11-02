# EEG-Agent desktop installer for Windows (PowerShell 5.1+).
#
#   irm https://raw.githubusercontent.com/berdakh/EEG-Source-Imaging-with-MNE-Python/master/installers/install.ps1 | iex
#
# Finds Python 3.10+, creates %USERPROFILE%\.eegagent\venv, installs the package with the
# desktop extras, and adds Start Menu and Desktop shortcuts. Re-run to upgrade.
$ErrorActionPreference = "Stop"

$AppDir = if ($env:EEGAGENT_HOME) { $env:EEGAGENT_HOME } else { Join-Path $env:USERPROFILE ".eegagent" }
$Venv = Join-Path $AppDir "venv"
$Source = if ($env:EEGAGENT_SOURCE) { $env:EEGAGENT_SOURCE } else { "git+https://github.com/berdakh/EEG-Source-Imaging-with-MNE-Python.git@master" }

function Find-Python {
    foreach ($cmd in @("py -3.13", "py -3.12", "py -3.11", "py -3.10", "python3", "python")) {
        try {
            $parts = $cmd.Split(" ")
            $v = & $parts[0] $parts[1..($parts.Length)] -c "import sys; print('%d.%d' % sys.version_info[:2]); sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return $cmd }
        } catch {}
    }
    return $null
}

$Py = Find-Python
if (-not $Py) {
    Write-Error "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and re-run."
}
Write-Host "Using $Py"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
if (-not (Test-Path $Venv)) {
    $parts = $Py.Split(" ")
    & $parts[0] $parts[1..($parts.Length)] -m venv $Venv
}
$VPy = Join-Path $Venv "Scripts\python.exe"
& $VPy -m pip install --quiet --upgrade pip
Write-Host "Installing EEG-Agent from $Source (downloads MNE, SciPy and Streamlit; a few minutes)..."
if (Test-Path $Source) {
    & $VPy -m pip install --quiet --upgrade "$Source[desktop]"
} else {
    & $VPy -m pip install --quiet --upgrade "eeg-source-imaging[desktop] @ $Source"
}

$Launcher = Join-Path $Venv "Scripts\eegagent-desktop.exe"
$Shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
    $lnk = $Shell.CreateShortcut((Join-Path $dir "EEG-Agent.lnk"))
    $lnk.TargetPath = $Launcher
    $lnk.WorkingDirectory = $AppDir
    $lnk.Description = "EEG source imaging by conversation"
    $lnk.Save()
}
Write-Host "Installed. Launch 'EEG-Agent' from the Start Menu or the Desktop shortcut."
Write-Host "Settings (provider, API key, Ollama host) are in the app's sidebar and saved to $AppDir\config.json."
