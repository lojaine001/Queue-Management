Option Explicit
Dim shell, files, scriptDir, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
scriptDir = files.GetParentFolderName(WScript.ScriptFullName)
command = "cmd /c " & Chr(34) & files.BuildPath(scriptDir, "run_ensemble.bat") & Chr(34)
exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode
