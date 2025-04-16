{******************************************************************************}
{                       CnPack For Delphi/C++Builder                           }
{                     �й����Լ��Ŀ���Դ�������������                         }
{                   (C)Copyright 2001-2025 CnPack ������                       }
{                   ------------------------------------                       }
{                                                                              }
{            ���������ǿ�Դ��������������������� CnPack �ķ���Э������        }
{        �ĺ����·�����һ����                                                }
{                                                                              }
{            ������һ��������Ŀ����ϣ�������ã���û���κε���������û��        }
{        �ʺ��ض�Ŀ�Ķ������ĵ���������ϸ���������� CnPack ����Э�顣        }
{                                                                              }
{            ��Ӧ���Ѿ��Ϳ�����һ���յ�һ�� CnPack ����Э��ĸ��������        }
{        ��û�У��ɷ������ǵ���վ��                                            }
{                                                                              }
{            ��վ��ַ��https://www.cnpack.org                                  }
{            �����ʼ���master@cnpack.org                                       }
{                                                                              }
{******************************************************************************}

program CnDebugViewer64;

{$IFNDEF WIN64}
  {$MESSAGE ERROR 'CnDebugViewer64 Compiled Only by Delphi 64-Bit Compiler.'}
{$ENDIF}

uses
  SysUtils,
  Forms,
  CnViewMain in 'CnViewMain.pas' {CnMainViewer},
  CnDebugIntf in 'CnDebugIntf.pas',
  CnMsgClasses in 'CnMsgClasses.pas',
  CnGetThread in 'CnGetThread.pas',
  CnViewCore in 'CnViewCore.pas',
  CnMdiView in 'CnMdiView.pas' {CnMsgChild},
  CnMsgFiler in 'CnMsgFiler.pas',
  CnFilterFrm in 'CnFilterFrm.pas' {CnSenderFilterFrm},
  CnViewOption in 'CnViewOption.pas' {CnViewerOptionsFrm},
  CnWatchFrm in 'CnWatchFrm.pas' {CnWatchForm},
  CnWizCfgUtils in '..\..\Source\Utils\CnWizCfgUtils.pas';

{$R *.RES}
{$R ..\WindowsXP.RES}

begin
  if GetCWUseCustomUserDir then
    LoadOptions(GetCWUserPath + SCnOptionFileName)
  else
    LoadOptions(ExtractFilePath(Application.ExeName) + SCnOptionFileName);

  if FindCmdLineSwitch('global', ['-', '/'], True) then
  begin
    // Global Switch first, using Global Mode
    IsLocalMode := False;
  end
  else if CnViewerOptions.LocalSession or FindCmdLineSwitch('local', ['-', '/'], True) then
  begin
    ReInitLocalConsts; // If Local Switch or Settings, using Local Mode
    IsLocalMode := True;
  end;

  if CheckRunning then Exit;
  Application.Initialize;
  Application.CreateForm(TCnMainViewer, CnMainViewer);
  CnMainViewer.LaunchThread;
  Application.Run;
end.
