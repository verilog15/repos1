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

unit CnScript_CnWizShortCut;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��ű��� CnWizShortCut ע����
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע���õ�Ԫ�� UnitParser v0.7 �Զ����ɵ��ļ��޸Ķ���
* ����ƽ̨��PWinXP SP2 + Delphi 7.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����ô����е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2017.05.18 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
   SysUtils
  ,Classes
  ,uPSComponent
  ,uPSRuntime
  ,uPSCompiler
  ;

type
(*----------------------------------------------------------------------------*)
  TPSImport_CnWizShortCut = class(TPSPlugin)
  protected
    procedure CompileImport1(CompExec: TPSScript); override;
    procedure ExecImport1(CompExec: TPSScript; const ri: TPSRuntimeClassImporter); override;
  end;


{ compile-time registration functions }
procedure SIRegister_TCnWizShortCutMgr(CL: TPSPascalCompiler);
procedure SIRegister_TCnWizShortCut(CL: TPSPascalCompiler);
procedure SIRegister_CnWizShortCut(CL: TPSPascalCompiler);

{ run-time registration functions }
procedure RIRegister_CnWizShortCut_Routines(S: TPSExec);
procedure RIRegister_TCnWizShortCutMgr(CL: TPSRuntimeClassImporter);
procedure RIRegister_TCnWizShortCut(CL: TPSRuntimeClassImporter);
procedure RIRegister_CnWizShortCut(CL: TPSRuntimeClassImporter);

procedure Register;

implementation

uses
   Windows
  ,Messages
  ,Menus
  ,ExtCtrls
  ,ToolsAPI
  ,CnWizConsts
  ,CnCommon
  ,CnWizShortCut
  ;

procedure Register;
begin
  RegisterComponents('Pascal Script', [TPSImport_CnWizShortCut]);
end;

(* === compile-time registration functions === *)
(*----------------------------------------------------------------------------*)
procedure SIRegister_TCnWizShortCutMgr(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TObject', 'TCnWizShortCutMgr') do
  with CL.AddClassN(CL.FindClass('TObject'),'TCnWizShortCutMgr') do
  begin
    RegisterMethod('Constructor Create');
    RegisterMethod('Function IndexOfShortCut( AWizShortCut : TCnWizShortCut) : Integer');
    RegisterMethod('Function IndexOfName( const AName : string) : Integer');
    RegisterMethod('Function Add( const AName : string; AShortCut : TShortCut; AKeyProc : TNotifyEvent; const AMenuName : string; ATag : Integer) : TCnWizShortCut');
    RegisterMethod('Procedure Delete( Index : Integer)');
    RegisterMethod('Procedure DeleteShortCut( var AWizShortCut : TCnWizShortCut)');
    RegisterMethod('Procedure Clear');
    RegisterMethod('Procedure BeginUpdate');
    RegisterMethod('Procedure EndUpdate');
    RegisterMethod('Function Updating : Boolean');
    RegisterMethod('Procedure UpdateBinding');
    RegisterProperty('Count', 'Integer', iptr);
    RegisterProperty('ShortCuts', 'TCnWizShortCut Integer', iptr);
  end;
end;

(*----------------------------------------------------------------------------*)
procedure SIRegister_TCnKeyBinding(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TNotifierObject', 'TCnKeyBinding') do
  with CL.AddClassN(CL.FindClass('TNotifierObject'),'TCnKeyBinding') do
  begin
    RegisterMethod('Constructor Create( AOwner : TCnWizShortCutMgr)');
    RegisterMethod('Function GetBindingType : TBindingType');
    RegisterMethod('Function GetDisplayName : string');
    RegisterMethod('Function GetName : string');
    RegisterMethod('Procedure BindKeyboard( const BindingServices : IOTAKeyBindingServices)');
  end;
end;

(*----------------------------------------------------------------------------*)
procedure SIRegister_TCnWizShortCut(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TObject', 'TCnWizShortCut') do
  with CL.AddClassN(CL.FindClass('TObject'),'TCnWizShortCut') do
  begin
    RegisterMethod('Constructor Create( AOwner : TCnWizShortCutMgr; const AName : string; AShortCut : TShortCut; AKeyProc : TNotifyEvent; const AMenuName : string; ATag : Integer)');
    RegisterProperty('Name', 'string', iptr);
    RegisterProperty('ShortCut', 'TShortCut', iptrw);
    RegisterProperty('KeyProc', 'TNotifyEvent', iptrw);
    RegisterProperty('MenuName', 'string', iptrw);
    RegisterProperty('Tag', 'Integer', iptrw);
  end;
end;

(*----------------------------------------------------------------------------*)
procedure SIRegister_CnWizShortCut(CL: TPSPascalCompiler);
begin
  CL.AddClassN(CL.FindClass('TOBJECT'),'TCnWizShortCutMgr');
  SIRegister_TCnWizShortCut(CL);
  SIRegister_TCnKeyBinding(CL);
  SIRegister_TCnWizShortCutMgr(CL);
  CL.AddDelphiFunction('Function WizShortCutMgr : TCnWizShortCutMgr');
  // CL.AddDelphiFunction('Procedure FreeWizShortCutMgr');
end;

(* === run-time registration functions === *)
(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutMgrShortCuts_R(Self: TCnWizShortCutMgr; var T: TCnWizShortCut; const t1: Integer);
begin T := Self.ShortCuts[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutMgrCount_R(Self: TCnWizShortCutMgr; var T: Integer);
begin T := Self.Count; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutTag_W(Self: TCnWizShortCut; const T: Integer);
begin Self.Tag := T; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutTag_R(Self: TCnWizShortCut; var T: Integer);
begin T := Self.Tag; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutMenuName_W(Self: TCnWizShortCut; const T: string);
begin Self.MenuName := T; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutMenuName_R(Self: TCnWizShortCut; var T: string);
begin T := Self.MenuName; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutKeyProc_W(Self: TCnWizShortCut; const T: TNotifyEvent);
begin Self.KeyProc := T; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutKeyProc_R(Self: TCnWizShortCut; var T: TNotifyEvent);
begin T := Self.KeyProc; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutShortCut_W(Self: TCnWizShortCut; const T: TShortCut);
begin Self.ShortCut := T; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutShortCut_R(Self: TCnWizShortCut; var T: TShortCut);
begin T := Self.ShortCut; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizShortCutName_R(Self: TCnWizShortCut; var T: string);
begin T := Self.Name; end;

(*----------------------------------------------------------------------------*)
procedure RIRegister_CnWizShortCut_Routines(S: TPSExec);
begin
  S.RegisterDelphiFunction(@WizShortCutMgr, 'WizShortCutMgr', cdRegister);
  // S.RegisterDelphiFunction(@FreeWizShortCutMgr, 'FreeWizShortCutMgr', cdRegister);
end;

(*----------------------------------------------------------------------------*)
procedure RIRegister_TCnWizShortCutMgr(CL: TPSRuntimeClassImporter);
begin
  with CL.Add(TCnWizShortCutMgr) do
  begin
    RegisterConstructor(@TCnWizShortCutMgr.Create, 'Create');
    RegisterMethod(@TCnWizShortCutMgr.IndexOfShortCut, 'IndexOfShortCut');
    RegisterMethod(@TCnWizShortCutMgr.IndexOfName, 'IndexOfName');
    RegisterMethod(@TCnWizShortCutMgr.Add, 'Add');
    RegisterMethod(@TCnWizShortCutMgr.Delete, 'Delete');
    RegisterMethod(@TCnWizShortCutMgr.DeleteShortCut, 'DeleteShortCut');
    RegisterMethod(@TCnWizShortCutMgr.Clear, 'Clear');
    RegisterMethod(@TCnWizShortCutMgr.BeginUpdate, 'BeginUpdate');
    RegisterMethod(@TCnWizShortCutMgr.EndUpdate, 'EndUpdate');
    RegisterMethod(@TCnWizShortCutMgr.Updating, 'Updating');
    RegisterMethod(@TCnWizShortCutMgr.UpdateBinding, 'UpdateBinding');
    RegisterPropertyHelper(@TCnWizShortCutMgrCount_R,nil,'Count');
    RegisterPropertyHelper(@TCnWizShortCutMgrShortCuts_R,nil,'ShortCuts');
  end;
end;

(*----------------------------------------------------------------------------*)
procedure RIRegister_TCnWizShortCut(CL: TPSRuntimeClassImporter);
begin
  with CL.Add(TCnWizShortCut) do
  begin
    RegisterConstructor(@TCnWizShortCut.Create, 'Create');
    RegisterPropertyHelper(@TCnWizShortCutName_R,nil,'Name');
    RegisterPropertyHelper(@TCnWizShortCutShortCut_R,@TCnWizShortCutShortCut_W,'ShortCut');
    RegisterPropertyHelper(@TCnWizShortCutKeyProc_R,@TCnWizShortCutKeyProc_W,'KeyProc');
    RegisterPropertyHelper(@TCnWizShortCutMenuName_R,@TCnWizShortCutMenuName_W,'MenuName');
    RegisterPropertyHelper(@TCnWizShortCutTag_R,@TCnWizShortCutTag_W,'Tag');
  end;
end;

(*----------------------------------------------------------------------------*)
procedure RIRegister_CnWizShortCut(CL: TPSRuntimeClassImporter);
begin
  with CL.Add(TCnWizShortCutMgr) do
  RIRegister_TCnWizShortCut(CL);
  RIRegister_TCnWizShortCutMgr(CL);
end;

{ TPSImport_CnWizShortCut }
(*----------------------------------------------------------------------------*)
procedure TPSImport_CnWizShortCut.CompileImport1(CompExec: TPSScript);
begin
  SIRegister_CnWizShortCut(CompExec.Comp);
end;
(*----------------------------------------------------------------------------*)
procedure TPSImport_CnWizShortCut.ExecImport1(CompExec: TPSScript; const ri: TPSRuntimeClassImporter);
begin
  RIRegister_CnWizShortCut(ri);
  RIRegister_CnWizShortCut_Routines(CompExec.Exec); // comment it if no routines
end;
(*----------------------------------------------------------------------------*)

end.
