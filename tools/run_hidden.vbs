' Runs a PowerShell script with no visible window (used by the IQMS scheduled tasks
' so nothing flashes on the store PC). Usage: wscript run_hidden.vbs <script.ps1> [args]
Option Explicit
Dim shell, args, i, command
Set shell = CreateObject("WScript.Shell")
Set args = WScript.Arguments
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File " & Chr(34) & args(0) & Chr(34)
For i = 1 To args.Count - 1
    command = command & " " & args(i)
Next
WScript.Quit shell.Run(command, 0, True)
