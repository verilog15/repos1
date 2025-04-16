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

unit CnWizDllEntry;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�CnWizard ר�� DLL ��ڵ�Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2018.08.27 V1.1
*               ���ӿ����������� AddWizard
*           2002.12.07 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  SysUtils, ToolsAPI, CnWizConsts;

// ר�� DLL ��ʼ����ں���
function InitWizard(const BorlandIDEServices: IBorlandIDEServices;
  RegisterProc: TWizardRegisterProc;
  var Terminate: TWizardTerminateProc): Boolean; stdcall;
{* ר�� DLL ��ʼ����ں���}

exports
  InitWizard name WizardEntryPoint;

implementation

uses
{$IFDEF DEBUG}
  CnDebug,
{$ENDIF}
  CnWizManager;

const
  InvalidIndex = -1;

var
  FWizardIndex: Integer = InvalidIndex;

// ר�� DLL �ͷŹ���
procedure FinalizeWizard;
var
  WizardServices: IOTAWizardServices;
begin
  if (FWizardIndex <> InvalidIndex) and (TObject(CnWizardMgr) is TInterfacedObject) then
  begin
    Assert(Assigned(BorlandIDEServices));
    WizardServices := BorlandIDEServices as IOTAWizardServices;
    Assert(Assigned(WizardServices));
{$IFDEF DEBUG}
    CnDebugger.LogMsg('CnWizardMgr Remove at ' + IntToStr(FWizardIndex));
{$ENDIF}
    WizardServices.RemoveWizard(FWizardIndex);
    FWizardIndex := InvalidIndex;
  end
  else
  begin
    FreeAndNil(CnWizardMgr);
{$IFDEF DEBUG}
    CnDebugger.LogMsg('Manually Free CnWizardMgr');
{$ENDIF}
  end;
end;

// ר�� DLL ��ʼ����ں���
function InitWizard(const BorlandIDEServices: IBorlandIDEServices;
  RegisterProc: TWizardRegisterProc;
  var Terminate: TWizardTerminateProc): Boolean; stdcall;
var
  WizardServices: IOTAWizardServices;
  AWizard: IOTAWizard;
  Reg: Boolean;
begin
  if FindCmdLineSwitch(SCnNoServiceCnWizardsSwitch, ['/', '-'], True) then
  begin
    Reg := False;
{$IFDEF DEBUG}
    CnDebugger.LogMsg('Create but Do NOT Register CnWizards');
{$ENDIF}
  end
  else
    Reg := True;

{$IFDEF DEBUG}
  CnDebugger.StartTimeMark('CWS');  // CnWizards Start-Up Timing Start
  CnDebugger.LogMsg('Wizard Dll Entry');
{$ENDIF}

  Result := BorlandIDEServices <> nil;
  if Result then
  begin
    Assert(ToolsAPI.BorlandIDEServices = BorlandIDEServices);
    Terminate := FinalizeWizard;
    WizardServices := BorlandIDEServices as IOTAWizardServices;
    Assert(Assigned(WizardServices));

    CnWizardMgr := TCnWizardMgr.Create;
    if Reg and Supports(TObject(CnWizardMgr), IOTAWizard, AWizard) then
    begin
      // ֻ�������в�Ҫ��ע�ᣬ�� CnWizardMgr ֧�� IOTAWizard �ӿڣ���ע��
      FWizardIndex := WizardServices.AddWizard(AWizard);
      Result := (FWizardIndex >= 0);
{$IFDEF DEBUG}
      CnDebugger.LogBoolean(Result, 'CnWizardMgr Registered at ' + IntToStr(FWizardIndex));
{$ENDIF}
    end
    else
    begin
      Result := True;
{$IFDEF DEBUG}
      CnDebugger.LogBoolean(Result, 'CnWizardMgr Created');
{$ENDIF}
    end;
  end;
end;

initialization
{$IFDEF DEBUG}
  if CnDebugger.ExceptTracking then
    CnDebugger.LogMsg('DllEntry initialization. CaptureStack Enabled')
  else
    CnDebugger.LogMsg('DllEntry initialization. CaptureStack Disabled')
{$ENDIF}

end.

