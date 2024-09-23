Attribute VB_Name = "Module1"
'PPM Image Viewer for Visual Basic 6.0
'
'Copyright 2003 by Dmitry Brant, All Rights Reserved.

'Version 1.1, Last modified 04-27-2000
'
'me@dmitrybrant.com
'
'http://www.dmitrybrant.com


'Feel free to make improvements or optimizations to the program.
'If you do, please tell me or show me what you have done.

'This program is FREEWARE. Use freely, but give me credit
'where credit is due. (this program is NOT in the
'public domain). If you will use this source code in
'your software, the copyright notice stated above
'must be included in your source code and/or in the
'Help/About box of your program.

'The SOURCE CODE of this program, or portions of it, may not
'be redistributed, either by itself or in a compilation
'package, without express authorization from the author.

'ALL PRODUCT NAMES MENTIONED IN THIS SOFTWARE ARE TRADEMARKS
'OR REGISTERED TRADEMARKS OF THEIR RESPECTIVE OWNERS.

'THIS SOFTWARE IS DISTRIBUTED "AS IS". THE SOFTWARE IS
'DISTRIBUTED WITH NO WARRANTY, EIHER EXPRESSED
'OR IMPLIED, INCLUDING, BY WAY OF EXAMPLE THE IMPLIED
'WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR
'PURPOSE. THE AUTHOR WILL NOT BE LIABLE FOR DATA LOSS, DAMAGES,
'LOSS OF PROFITS OR ANY OTHER KIND OF LOSS RESULTING FROM THE
'USE OR MISUSE OF THIS SOFTWARE.

Option Explicit
DefInt A-Z

Declare Function SetPixel Lib "gdi32" (ByVal hdc As Long, ByVal x As Long, ByVal y As Long, ByVal crColor As Long) As Long

Sub DoPPM(FName As String, FObject As Object)
    Dim FWidth As Long, FHeight As Long
    Dim FColors As Integer, dx As Long, dy As Long
    Dim str1 As String, FType As String, theHdc As Long
    Dim r As Integer, g As Integer, b As Integer
    
    On Error GoTo ErrorTrap0
    
    Open FName For Binary As #1
    
    dx = 0: dy = 0
    
    FType = GetNextObj
    If FType <> "P3" And FType <> "P6" Then
        Close #1
        MsgBox "This is not a valid PPM File.", , "Error"
        Exit Sub
    End If
    
    Do
        str1 = GetNextObj
        If Left(str1, 1) <> "#" Then Exit Do
    Loop
    
    FWidth = Val(str1)
    FHeight = Val(GetNextObj)
    FColors = Val(GetNextObj)
    
    theHdc = FObject.hdc
    If FType = "P3" Then
        For dy = 0 To FHeight - 1
            For dx = 0 To FWidth - 1
                r = Val(GetNextObj)
                g = Val(GetNextObj)
                b = Val(GetNextObj)
                SetPixel theHdc, dx, dy, RGB(r, g, b)
            Next dx
            If dy Mod 16 = 0 Then FObject.Refresh: DoEvents
        Next dy
        Close #1
    Else
        ReDim Scan(0 To FWidth * FHeight * 3 - 1) As Byte
        Get #1, , Scan
        Close #1
        For dy = 0 To FHeight - 1
            For dx = 0 To FWidth - 1
                r = Scan(3 * (dx + dy * FWidth))
                g = Scan(3 * (dx + dy * FWidth) + 1)
                b = Scan(3 * (dx + dy * FWidth) + 2)
                SetPixel theHdc, dx, dy, RGB(r, g, b)
            Next dx
            If dy Mod 32 = 0 Then FObject.Refresh: DoEvents
        Next dy
        Erase Scan
    End If
    Exit Sub

ErrorTrap0:
    Close #1
    MsgBox "There was an error reading the file:" + vbCrLf + Error$, vbCritical, "Error"
End Sub

Function GetNextObj() As String
    Dim a As Byte, Res As String
    
    'eliminate whitespace
    Do
        Get #1, , a
    Loop Until a > 32 Or EOF(1)
    
    
    If a = 35 Then     'it's a comment
        Do Until a = 13 Or a = 10
            Res = Res + Chr(a)
            Get #1, , a
        Loop
        GetNextObj = Res
        Exit Function
    End If

    Do Until a <= 32
        Res = Res + Chr(a)
        Get #1, , a
    Loop
    GetNextObj = Res
End Function
