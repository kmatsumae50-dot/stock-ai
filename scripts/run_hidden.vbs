' scripts/run_hidden.vbs
' Runs a target command completely hidden (SW_HIDE = 0) without creating any console window
' or stealing active window focus, and propagates the exit code.
Option Explicit

Dim objShell, objArgs, strCommand, i, strArg, intExitCode

Set objShell = CreateObject("WScript.Shell")
Set objArgs = WScript.Arguments

If objArgs.Count = 0 Then
    WScript.Quit 1
End If

strCommand = ""
For i = 0 To objArgs.Count - 1
    strArg = objArgs(i)
    ' Quote argument if it contains spaces and is not already wrapped in double quotes
    If InStr(strArg, " ") > 0 Then
        If Left(strArg, 1) <> """" Or Right(strArg, 1) <> """" Then
            strArg = """" & strArg & """"
        End If
    End If
    
    If strCommand = "" Then
        strCommand = strArg
    Else
        strCommand = strCommand & " " & strArg
    End If
Next

intExitCode = objShell.Run(strCommand, 0, True)
WScript.Quit intExitCode
