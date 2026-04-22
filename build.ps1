param(
    [switch]$InstallFullDataset = $false,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==== $msg ====" -ForegroundColor Cyan
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

$DataDir = Join-Path $ProjectRoot "data"
$GFRepoDir = Join-Path $DataDir "graspfactory_repo"
$GFDatasetDir = Join-Path $DataDir "graspfactory"

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path $GFDatasetDir | Out-Null

Write-Step "Creating virtual environment"
if (!(Test-Path $VenvPython)) {
    & $PythonExe -m venv $VenvPath
}
else {
    Write-Host "Virtual environment already exists at $VenvPath"
}

Write-Step "Upgrading pip / wheel / setuptools"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Installing project requirements"
& $VenvPip install -r requirements.txt

Write-Step "Cloning GraspFactory repository if needed"
if (!(Test-Path $GFRepoDir)) {
    git clone https://github.com/AutodeskRoboticsLab/graspfactory.git $GFRepoDir
}
else {
    Write-Host "GraspFactory repo already exists at $GFRepoDir"
}

Write-Step "Installing GraspFactory repo requirements"
$GFRequirements = Join-Path $GFRepoDir "requirements.txt"
if (Test-Path $GFRequirements) {
    & $VenvPip install -r $GFRequirements
}
else {
    Write-Warning "Could not find GraspFactory requirements.txt"
}

Write-Step "Copying sample dataset into project data folder"
$SampleSource = Join-Path $GFRepoDir "sample_data"
if (Test-Path $SampleSource) {
    Copy-Item -Path (Join-Path $SampleSource "*") -Destination $GFDatasetDir -Recurse -Force
    Write-Host "Sample data copied to $GFDatasetDir"
}
else {
    Write-Warning "sample_data folder not found in GraspFactory repo"
}

if ($InstallFullDataset) {
    Write-Step "Attempting full dataset download"
    Write-Host "Looking for a download script inside the GraspFactory repo..."

    $CandidateScripts = @(
        (Join-Path $GFRepoDir "download_data.py"),
        (Join-Path $GFRepoDir "scripts\download_data.py"),
        (Join-Path $GFRepoDir "download_dataset.py"),
        (Join-Path $GFRepoDir "scripts\download_dataset.py")
    ) | Where-Object { Test-Path $_ }

    if ($CandidateScripts.Count -gt 0) {
        $DownloadScript = $CandidateScripts[0]
        Write-Host "Using: $DownloadScript"

        Push-Location $GFRepoDir
        try {
            & $VenvPython $DownloadScript
        }
        finally {
            Pop-Location
        }

        Write-Host "Full dataset download command completed."
        Write-Host "Check the repo output folders and move / link the Robotiq 2F-85 subset into data\graspfactory if needed."
    }
    else {
        Write-Warning "No recognized dataset download script was found automatically."
        Write-Warning "Open the GraspFactory repo README and run its current dataset download command manually from:"
        Write-Warning "  $GFRepoDir"
    }
}

Write-Step "Build complete"
Write-Host "Virtual environment: $VenvPath"
Write-Host "Project dataset dir: $GFDatasetDir"
Write-Host ""
Write-Host "To activate the environment in PowerShell:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run your script:"
Write-Host "  .\.venv\Scripts\python.exe .\src\main.py"