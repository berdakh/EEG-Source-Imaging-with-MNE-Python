# Install day script for Windows. Run from the repo root in PowerShell.
$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host "Install Ollama from https://ollama.com/download/windows then re-run this script."
  exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv..."
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv sync --all-extras
if (Get-Command npm -ErrorAction SilentlyContinue) {
  Push-Location frontend; npm install; npm run build; Pop-Location
} else {
  Write-Host "npm not found; frontend not built. Install Node 22 and run: cd frontend; npm install; npm run build"
}

uv run python scripts/pull_models.py
$data = if ($env:LOA_DATA_DIR) { $env:LOA_DATA_DIR } else { "$env:LOCALAPPDATA\LocalOfficeAI" }
New-Item -ItemType Directory -Force -Path $data | Out-Null
Write-Host "`nDone. Start with: desktop\launcher\start.bat"
