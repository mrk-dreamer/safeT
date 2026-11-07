; Inno Setup 6 script for SAFE-T
; Built by packaging/build_bundle.ps1 -Installer

#ifndef SourceDir
  #define SourceDir "..\dist\SAFE-T"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#define AppName "SAFE-T"
#define AppVersion "1.0.0"
#define AppPublisher "SAFE-T"
#define AppURL "https://github.com/mrk-dreamer/safeT"
#define AppExeName "SAFE-T.exe"

[Setup]
AppId={{8D6D8F61-3C4A-4B79-9E56-2A9A711D0491}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
; Install per-user under %LOCALAPPDATA%\Programs\SAFE-T so no admin
; prompt is needed and the application directory is always writable.
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SAFE-T-Setup
Compression=lzma2/fast
SolidCompression=no
; Enable disk spanning in case the bundled PyTorch + MFA payload exceeds 2 GB
DiskSpanning=yes
DiskSliceSize=2100000000
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
LicenseFile=..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Unpack and relocate the bundled MFA environment during installation so the first
; alignment run in the GUI starts immediately.
Filename: "{app}\{#AppExeName}"; Parameters: "--prepare-mfa"; WorkingDir: "{app}"; \
    StatusMsg: "Preparing Montreal Forced Aligner environment (first-time setup)..."; \
    Flags: runhidden waituntilterminated
Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; \
    Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up runtime output files created inside the application directory
Type: filesandordirs; Name: "{app}\PyPraat_sentences"
Type: files; Name: "{app}\*.log"
Type: files; Name: "{app}\*.txt"
Type: files; Name: "{app}\*.json"
Type: files; Name: "{app}\formant_targets.csv"
Type: files; Name: "{app}\formants_values.csv"
