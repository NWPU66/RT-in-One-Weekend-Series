VERSION 5.00
Object = "{F9043C88-F6F2-101A-A3C9-08002B2F49FB}#1.2#0"; "comdlg32.ocx"
Begin VB.Form Form1 
   AutoRedraw      =   -1  'True
   BackColor       =   &H8000000C&
   Caption         =   "PPMView 1.1 by Dmitry Brant"
   ClientHeight    =   4665
   ClientLeft      =   165
   ClientTop       =   765
   ClientWidth     =   6615
   LinkTopic       =   "Form1"
   ScaleHeight     =   311
   ScaleMode       =   3  'Pixel
   ScaleWidth      =   441
   StartUpPosition =   3  'Windows Default
   Begin MSComDlg.CommonDialog CommonDialog1 
      Left            =   0
      Top             =   0
      _ExtentX        =   847
      _ExtentY        =   847
      _Version        =   393216
      CancelError     =   -1  'True
   End
   Begin VB.Menu mnuFile 
      Caption         =   "&File"
      Begin VB.Menu mnuOpen 
         Caption         =   "&Open"
      End
      Begin VB.Menu mnuSep1 
         Caption         =   "-"
      End
      Begin VB.Menu mnuExit 
         Caption         =   "&Exit"
      End
   End
   Begin VB.Menu mnuHelp 
      Caption         =   "&Help"
      Begin VB.Menu mnuAbout 
         Caption         =   "&About"
      End
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
'For Copyright information, read the comments
'in the Module file (.BAS)

Option Explicit

Private Sub Form_Unload(Cancel As Integer)
    Close
    End
End Sub

Private Sub mnuAbout_Click()
    MsgBox "PPMView 1.0 by Dmitry Brant (me@dmitrybrant.com)" + vbCrLf + vbCrLf + _
            "A complete PPM decoder written entirely in VB." + vbCrLf + vbCrLf + _
            "Copyright 2003 by Dmitry Brant, All Rights Reserved." + vbCrLf + vbCrLf + _
            "http://www.dmitrybrant.com", vbInformation, "About JpgView"
End Sub

Private Sub mnuExit_Click()
    Close
    End
End Sub

Private Sub mnuOpen_Click()
Dim sFilter As String
    On Error GoTo ErrHandler
    CommonDialog1.DialogTitle = "Select a PPM Image"
    sFilter = "PPM Images (*.ppm)|*.ppm"
    sFilter = sFilter & "|All Files (*.*)|*.*"
    CommonDialog1.Filter = sFilter
    CommonDialog1.ShowOpen
    Form1.Refresh
    mnuOpen.Enabled = False
    Form1.Cls
    DoPPM CommonDialog1.FileName, Form1
    Form1.Refresh
    mnuOpen.Enabled = True
    Beep
ErrHandler:
' User pressed Cancel button.
End Sub
