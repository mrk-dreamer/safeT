<#
.SYNOPSIS
    Builds the self-contained SAFE-T bundle (and optionally a Setup.exe / zip).

.DESCRIPTION
    Produces dist\SAFE-T\ that runs on a clean Windows 10/11 x64 machine
    without Python, conda, MFA, VLC or FFmpeg installed:

        SAFE-T.exe, _internal\               PyInstaller one-folder build
        ffmpeg\                              ffmpeg.exe (+ ffprobe.exe)
        vlc\                                 libvlc.dll, libvlccore.dll, plugins\
        mfa_env.tar                          conda-pack archive of the MFA environment
        model_cache\                         pre-downloaded WhisperX / wav2vec2 / KittenTTS models
        pretrained_models\, *.csv, *.onnx    data files read from the working directory
        mfa_align.py                         run by the MFA environment's Python

    Build machine requirements: the dev setup from README.md (.venv with
    requirements.txt + requirements_build.txt, the MFA conda env, VLC, FFmpeg,
    MFA models and Piper voices) plus `conda install -n base -c conda-forge conda-pack`.
    Inno Setup 6 is only needed for -Installer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build_bundle.ps1 -Installer
#>
[CmdletBinding()]
param(
    [string]$MfaEnvName = "mfa_env1",
    [string]$Python = "",
    [string]$Conda = "",
    [string]$VlcDir = "",
    [string]$FfmpegExe = "",
    [string]$CondaPackExtraArgs = "",
    [switch]$SkipPyInstaller,
    [switch]$SkipMfa,
    [switch]$SkipFfmpeg,
    [switch]$SkipVlc,
    [switch]$SkipCsv,
    [switch]$SkipPiperModels,
    [switch]$SkipMfaModels,
    [switch]$SkipHfTorchModels,
    [switch]$SkipModels,
    [switch]$CodeOnly,
    [switch]$Verify,
    [switch]$Installer,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Dist = Join-Path $ProjectRoot "dist\SAFE-T"

if ($CodeOnly) {
    $SkipMfa = $true
    $SkipFfmpeg = $true
    $SkipVlc = $true
    $SkipCsv = $true
    $SkipPiperModels = $true
    $SkipMfaModels = $true
    $SkipHfTorchModels = $true
}
if ($SkipModels) {
    $SkipPiperModels = $true
    $SkipMfaModels = $true
    $SkipHfTorchModels = $true
}

function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }
function Fail($message) { Write-Host "ERROR: $message" -ForegroundColor Red; exit 1 }
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    # Tools such as PyInstaller log to stderr; rely on the exit code instead.
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}
function First-Existing([string[]]$candidates) {
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path } }
    return $null
}

# ---------------------------------------------------------------- locate tools
Step "Checking build inputs"
if (-not $Python) { $Python = First-Existing @(".venv\Scripts\python.exe") }
if (-not $Python) { Fail "Python of the project venv not found. Pass -Python <path>." }

if (-not $SkipMfa) {
    if (-not $Conda) {
        $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
        $Conda = First-Existing @(
            $env:CONDA_EXE,
            $(if ($condaCmd) { $condaCmd.Source }),
            "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
            "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
            "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
            "$env:ProgramData\anaconda3\Scripts\conda.exe",
            "$env:ProgramData\miniconda3\Scripts\conda.exe"
        )
    }
    if (-not $Conda) { Fail "conda not found. Pass -Conda <path to conda.exe> or use -SkipMfa." }
}

if (-not $SkipVlc) {
    if (-not $VlcDir) {
        $VlcDir = First-Existing @("$env:ProgramFiles\VideoLAN\VLC", "${env:ProgramFiles(x86)}\VideoLAN\VLC")
    }
    if (-not $VlcDir -or -not (Test-Path (Join-Path $VlcDir "libvlc.dll"))) {
        Fail "VLC (64-bit) not found. Install it, pass -VlcDir <folder containing libvlc.dll>, or use -SkipVlc."
    }
}

if (-not $SkipFfmpeg) {
    if (-not $FfmpegExe) {
        $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
        $FfmpegExe = First-Existing @($(if ($ffmpegCmd) { $ffmpegCmd.Source }), "C:\ffmpeg\bin\ffmpeg.exe")
    }
    if (-not $FfmpegExe) { Fail "ffmpeg.exe not found. Pass -FfmpegExe <path> or use -SkipFfmpeg." }
}

$coreDataFiles = @(
    "mfa_align.py",
    "verify_all.py",
    "LICENSE"
)
$csvFiles = @(
    "landmark_identification_ground_truth.csv",
    "landmark_identification_phoneme_mapping.csv"
)
$piperModelFiles = @(
    "en_GB-alan-medium.onnx", "en_GB-alan-medium.onnx.json",
    "en_GB-alba-medium.onnx", "en_GB-alba-medium.onnx.json"
)
$mfaModelFiles = @(
    "pretrained_models\acoustic\english_mfa.zip",
    "pretrained_models\dictionary\english_uk_mfa.dict"
)

$dataFiles = @() + $coreDataFiles
if (-not $SkipCsv) { $dataFiles += $csvFiles }
if (-not $SkipPiperModels) { $dataFiles += $piperModelFiles }
if (-not $SkipMfaModels) { $dataFiles += $mfaModelFiles }

foreach ($f in $dataFiles) { if (-not (Test-Path $f)) { Fail "Required file missing: $f (see README.md)" } }

# Verify that any skipped items already exist in $Dist from a previous build.
$skippedExpectedInDist = @()
if ($SkipPyInstaller) { $skippedExpectedInDist += "SAFE-T.exe" }
if ($SkipMfa) { $skippedExpectedInDist += "mfa_env.tar" }
if ($SkipFfmpeg) { $skippedExpectedInDist += "ffmpeg\ffmpeg.exe" }
if ($SkipVlc) { $skippedExpectedInDist += @("vlc\libvlc.dll", "vlc\libvlccore.dll", "vlc\plugins") }
if ($SkipCsv) { $skippedExpectedInDist += $csvFiles }
if ($SkipPiperModels) { $skippedExpectedInDist += $piperModelFiles }
if ($SkipMfaModels) { $skippedExpectedInDist += $mfaModelFiles }
if ($SkipHfTorchModels) { $skippedExpectedInDist += "model_cache\.complete" }

foreach ($rel in $skippedExpectedInDist) {
    $existingPath = Join-Path $Dist $rel
    if (-not (Test-Path $existingPath)) {
        Fail "Skipped item not found in existing bundle: $existingPath. Run a build without skipping this item first."
    }
}

Write-Host "Python : $Python"
Write-Host "Conda  : $(if ($SkipMfa) { '(skipped)' } else { $Conda })"
Write-Host "VLC    : $(if ($SkipVlc) { '(skipped)' } else { $VlcDir })"
Write-Host "FFmpeg : $(if ($SkipFfmpeg) { '(skipped)' } else { $FfmpegExe })"

# ---------------------------------------------------------------- PyInstaller
if (-not $SkipPyInstaller) {
    Step "Building the application with PyInstaller (10-30 minutes)"
    Invoke-Checked $Python @("-m", "PyInstaller", "--version")
    $preserveDir = Join-Path $ProjectRoot "build\_preserve_bundle"
    if (Test-Path $preserveDir) { Remove-Item $preserveDir -Recurse -Force }
    New-Item -ItemType Directory -Force $preserveDir | Out-Null

    $preserveItems = @("model_cache")
    if ($SkipMfa) { $preserveItems += "mfa_env.tar" }
    if ($SkipFfmpeg) { $preserveItems += "ffmpeg" }
    if ($SkipVlc) { $preserveItems += "vlc" }
    if ($SkipCsv) { $preserveItems += $csvFiles }
    if ($SkipPiperModels) { $preserveItems += $piperModelFiles }
    if ($SkipMfaModels) { $preserveItems += "pretrained_models" }

    foreach ($relPath in $preserveItems) {
        $srcPath = Join-Path $Dist $relPath
        $dstPath = Join-Path $preserveDir $relPath
        if (-not (Test-Path $dstPath) -and (Test-Path $srcPath)) {
            New-Item -ItemType Directory -Force (Split-Path -Parent $dstPath) | Out-Null
            Move-Item $srcPath $dstPath -Force
        }
    }

    Invoke-Checked $Python @("-m", "PyInstaller", "--noconfirm",
        "--distpath", "dist", "--workpath", "build", "packaging\build_exe.spec")

    foreach ($relPath in $preserveItems) {
        $srcPath = Join-Path $preserveDir $relPath
        $dstPath = Join-Path $Dist $relPath
        if (Test-Path $srcPath) {
            New-Item -ItemType Directory -Force (Split-Path -Parent $dstPath) | Out-Null
            if (Test-Path $dstPath) { Remove-Item $dstPath -Recurse -Force }
            Move-Item $srcPath $dstPath -Force
        }
    }
    Remove-Item $preserveDir -Recurse -Force -ErrorAction SilentlyContinue
}
if (-not (Test-Path (Join-Path $Dist "SAFE-T.exe"))) { Fail "$Dist\SAFE-T.exe not found." }

# ---------------------------------------------------------------- data files
Step "Copying data files"
foreach ($f in $dataFiles) {
    $target = Join-Path $Dist $f
    New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
    Copy-Item $f $target -Force
}
Copy-Item "packaging\README_DISTRIBUTION.md" (Join-Path $Dist "README.md") -Force

# ---------------------------------------------------------------- FFmpeg
if (-not $SkipFfmpeg) {
    Step "Bundling FFmpeg"
    $ffmpegTarget = Join-Path $Dist "ffmpeg"
    New-Item -ItemType Directory -Force $ffmpegTarget | Out-Null
    Copy-Item $FfmpegExe $ffmpegTarget -Force
    $ffprobe = Join-Path (Split-Path -Parent $FfmpegExe) "ffprobe.exe"
    if (Test-Path $ffprobe) { Copy-Item $ffprobe $ffmpegTarget -Force }
    # Shared FFmpeg builds need their DLLs next to the executable.
    Get-ChildItem (Split-Path -Parent $FfmpegExe) -Filter "*.dll" -ErrorAction SilentlyContinue |
        Copy-Item -Destination $ffmpegTarget -Force
}

# ---------------------------------------------------------------- libVLC
if (-not $SkipVlc) {
    Step "Bundling libVLC"
    $vlcTarget = Join-Path $Dist "vlc"
    if (Test-Path $vlcTarget) { Remove-Item $vlcTarget -Recurse -Force }
    New-Item -ItemType Directory -Force $vlcTarget | Out-Null
    Copy-Item (Join-Path $VlcDir "libvlc.dll"), (Join-Path $VlcDir "libvlccore.dll") $vlcTarget -Force
    Copy-Item (Join-Path $VlcDir "plugins") $vlcTarget -Recurse -Force
    foreach ($legal in "COPYING.txt", "AUTHORS.txt") {
        $p = Join-Path $VlcDir $legal
        if (Test-Path $p) { Copy-Item $p (Join-Path $vlcTarget "VLC_$legal") -Force }
    }
}

# ---------------------------------------------------------------- MFA environment
if (-not $SkipMfa) {
    Step "Packing the '$MfaEnvName' conda environment with conda-pack"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"  # conda writes to stderr when the subcommand is unknown
    & $Conda pack --help *> $null
    $ErrorActionPreference = $previousPreference
    if ($LASTEXITCODE -ne 0) {
        Fail "conda-pack is not installed. Run: conda install -n base -c conda-forge conda-pack"
    }
    $archive = Join-Path $Dist "mfa_env.tar"
    Invoke-Checked $Conda @("run", "-n", "base", "--no-capture-output", "python", "packaging\pack_mfa_env.py", $MfaEnvName, $archive)
}

# ---------------------------------------------------------------- models
if (-not $SkipHfTorchModels) {
    Step "Pre-downloading runtime models"
    Invoke-Checked $Python @("packaging\prefetch_models.py", (Join-Path $Dist "model_cache"))
}

# ---------------------------------------------------------------- verify bundle completeness
Step "Verifying bundle contents in $Dist"
$allBundleRequired = @("SAFE-T.exe", "README.md", "mfa_env.tar", "ffmpeg\ffmpeg.exe", "vlc\libvlc.dll", "vlc\libvlccore.dll", "vlc\plugins", "model_cache\.complete") + $coreDataFiles + $csvFiles + $piperModelFiles + $mfaModelFiles
foreach ($rel in $allBundleRequired) {
    $full = Join-Path $Dist $rel
    if (-not (Test-Path $full)) {
        Fail "Bundle verification failed: missing '$full'."
    }
}
Write-Host "All $($allBundleRequired.Count) required bundle items verified present."

if ($Verify) {
    Step "Running functional verification suite on bundle"
    Invoke-Checked $Python @("verify_all.py", "--bundle", $Dist)
}

Step "Bundle ready: $Dist"
Write-Host "Test it by double-clicking $Dist\SAFE-T.exe"

# ---------------------------------------------------------------- packaging
if ($Zip) {
    Step "Creating zip archive"
    $zipPath = Join-Path $ProjectRoot "dist\SAFE-T.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    # bsdtar (built into Windows 10+) writes Zip64, unlike Compress-Archive.
    Invoke-Checked "tar.exe" @("-a", "-c", "-f", $zipPath, "-C", (Join-Path $ProjectRoot "dist"), "SAFE-T")
    Write-Host "Zip: $zipPath"
}

if ($Installer) {
    Step "Building the installer with Inno Setup"
    $isccCmd = Get-Command iscc -ErrorAction SilentlyContinue
    $iscc = First-Existing @(
        $(if ($isccCmd) { $isccCmd.Source }),
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    if (-not $iscc) { Fail "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isinfo.php" }
    Invoke-Checked $iscc @("/DSourceDir=$Dist", "/DOutputDir=$(Join-Path $ProjectRoot 'dist')", "packaging\installer.iss")
    Write-Host "Installer written to $(Join-Path $ProjectRoot 'dist')"
}
