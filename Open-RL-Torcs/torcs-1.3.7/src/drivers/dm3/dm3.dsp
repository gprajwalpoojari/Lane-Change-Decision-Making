# Microsoft Developer Studio Project File - Name="dm3" - Package Owner=<4>

# Microsoft Developer Studio Generated Build File, Format Version 6.00

# ** DO NOT EDIT **



# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102



CFG=dm3 - Win32 Debug

!MESSAGE This is not a valid makefile. To build this project using NMAKE,

!MESSAGE use the Export Makefile command and run

!MESSAGE 

!MESSAGE NMAKE /f "dm3.mak".

!MESSAGE 

!MESSAGE You can specify a configuration when running NMAKE

!MESSAGE by defining the macro CFG on the command line. For example:

!MESSAGE 

!MESSAGE NMAKE /f "dm3.mak" CFG="dm3 - Win32 Debug"

!MESSAGE 

!MESSAGE Possible choices for configuration are:

!MESSAGE 

!MESSAGE "dm3 - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")

!MESSAGE "dm3 - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")

!MESSAGE 



# Begin Project

# PROP AllowPerConfigDependencies 0

# PROP Scc_ProjName ""

# PROP Scc_LocalPath ""

CPP=cl.exe

MTL=midl.exe

RSC=rc.exe



!IF  "$(CFG)" == "dm3 - Win32 Release"



# PROP BASE Use_MFC 0

# PROP BASE Use_Debug_Libraries 0

# PROP BASE Output_Dir "Release"

# PROP BASE Intermediate_Dir "Release"

# PROP BASE Target_Dir ""

# PROP Use_MFC 0

# PROP Use_Debug_Libraries 0

# PROP Output_Dir "Release"

# PROP Intermediate_Dir "Release"

# PROP Ignore_Export_Lib 1

# PROP Target_Dir ""

# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "DM3_EXPORTS" /YX /FD /c

# ADD CPP /nologo /G5 /W3 /GX /O2 /I "../../../export/include" /I "../../windows/include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "DM3_EXPORTS" /YX /FD /c

# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32

# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32

# ADD BASE RSC /l 0x40c /d "NDEBUG"

# ADD RSC /l 0x40c /d "NDEBUG"

BSC32=bscmake.exe

# ADD BASE BSC32 /nologo

# ADD BSC32 /nologo

LINK32=link.exe

# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386

# ADD LINK32 tgf.lib robottools.lib sg.lib ul.lib /nologo /dll /map /machine:I386 /nodefaultlib:"LIBCD" /libpath:"../../../export/lib" /libpath:"../../windows/lib"

# Begin Special Build Tool

WkspDir=.

TargetDir=.\Release

SOURCE="$(InputPath)"

PostBuild_Cmds=copy $(TargetDir)\*.dll $(WkspDir)\runtime\drivers\dm3

# End Special Build Tool



!ELSEIF  "$(CFG)" == "dm3 - Win32 Debug"



# PROP BASE Use_MFC 0

# PROP BASE Use_Debug_Libraries 1

# PROP BASE Output_Dir "Debug"

# PROP BASE Intermediate_Dir "Debug"

# PROP BASE Target_Dir ""

# PROP Use_MFC 0

# PROP Use_Debug_Libraries 1

# PROP Output_Dir "Debug"

# PROP Intermediate_Dir "Debug"

# PROP Ignore_Export_Lib 1

# PROP Target_Dir ""

# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "DM3_EXPORTS" /YX /FD /GZ /c

# ADD CPP /nologo /G5 /W3 /Gm /GX /ZI /Od /I "../../../export/include" /I "../../windows/include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "DM3_EXPORTS" /YX /FD /GZ /c

# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32

# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32

# ADD BASE RSC /l 0x40c /d "_DEBUG"

# ADD RSC /l 0x40c /d "_DEBUG"

BSC32=bscmake.exe

# ADD BASE BSC32 /nologo

# ADD BSC32 /nologo

LINK32=link.exe

# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept

# ADD LINK32 robottools.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib sg.lib ul.lib /nologo /dll /map /debug /machine:I386 /pdbtype:sept /libpath:"../../../export/libd" /libpath:"../../windows/lib"

# Begin Special Build Tool

WkspDir=.

TargetDir=.\Debug

SOURCE="$(InputPath)"

PostBuild_Cmds=copy $(TargetDir)\*.dll $(WkspDir)\runtimed\drivers\dm3

# End Special Build Tool



!ENDIF 



# Begin Target



# Name "dm3 - Win32 Release"

# Name "dm3 - Win32 Debug"

# Begin Group "Source Files"



# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"

# Begin Source File



SOURCE=.\dm3.cpp

# End Source File

# Begin Source File



SOURCE=.\dm3.def

# End Source File

# End Group

# Begin Group "Header Files"



# PROP Default_Filter "h;hpp;hxx;hm;inl"

# End Group

# End Target

# End Project

