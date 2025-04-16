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

unit CnScript_CnWizManager;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��ű��� CnWizManager ע����
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע���õ�Ԫ�� UnitParser v0.7 �Զ����ɵ��ļ��޸Ķ���
* ����ƽ̨��PWinXP SP2 + Delphi 7.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����ô����е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2023.07.12 V1.0
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
  TPSImport_CnWizManager = class(TPSPlugin)
  protected
    procedure CompileImport1(CompExec: TPSScript); override;
    procedure ExecImport1(CompExec: TPSScript; const ri: TPSRuntimeClassImporter); override;
  end;


{ compile-time registration functions }
procedure SIRegister_TCnDesignSelectionManager(CL: TPSPascalCompiler);
procedure SIRegister_TCnWizardMgr(CL: TPSPascalCompiler);
procedure SIRegister_CnWizManager(CL: TPSPascalCompiler);

{ run-time registration functions }
procedure RIRegister_CnWizManager_Routines(S: TPSExec);
{$IFDEF COMPILER6_UP}
procedure RIRegister_TCnDesignSelectionManager(CL: TPSRuntimeClassImporter);
{$ENDIF}
procedure RIRegister_TCnWizardMgr(CL: TPSRuntimeClassImporter);
procedure RIRegister_CnWizManager(CL: TPSRuntimeClassImporter);

implementation


uses
   Windows
  ,Messages
  ,Graphics
  ,Controls
  ,Menus
  ,ActnList
  ,Forms
  ,ImgList
  ,ExtCtrls
  ,IniFiles
  ,Dialogs
  ,Registry
  ,ToolsAPI
  ,Contnrs
{$IFDEF COMPILER6_UP}
  ,DesignIntf
  ,DesignEditors
  ,DesignMenus
{$ELSE}
  ,DsgnIntf
{$ENDIF}
  ,CnWizClasses
  ,CnWizConsts
  ,CnWizMenuAction
  ,CnLangMgr
  ,CnRestoreSystemMenu
  ,CnWizIdeHooks
  ,CnWizManager
  ;


(* === compile-time registration functions === *)
(*----------------------------------------------------------------------------*)
procedure SIRegister_TCnDesignSelectionManager(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TBaseSelectionEditor', 'TCnDesignSelectionManager') do
  with CL.AddClassN(CL.FindClass('TBaseSelectionEditor'),'TCnDesignSelectionManager') do
  begin
    RegisterMethod('Procedure ExecuteVerb( Index : Integer; const List : IDesignerSelections)');
    RegisterMethod('Function GetVerb( Index : Integer) : string');
    RegisterMethod('Function GetVerbCount : Integer');
    RegisterMethod('Procedure PrepareItem( Index : Integer; const AItem : IMenuItem)');
    RegisterMethod('Procedure RequiresUnits( Proc : TGetStrProc)');
  end;
end;

(*----------------------------------------------------------------------------*)
procedure SIRegister_TCnWizardMgr(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TNotifierObject', 'TCnWizardMgr') do
  with CL.AddClassN(CL.FindClass('TNotifierObject'),'TCnWizardMgr') do
  begin
    RegisterMethod('Constructor Create');
    RegisterMethod('Function GetIDString : string');
    RegisterMethod('Function GetName : string');
    RegisterMethod('Function GetState : TWizardState');
    RegisterMethod('Procedure Execute');
    RegisterMethod('Procedure LoadSettings');
    RegisterMethod('Procedure SaveSettings');
    RegisterMethod('Procedure ConstructSortedMenu');
    RegisterMethod('Procedure UpdateMenuPos( UseToolsMenu : Boolean)');
    RegisterMethod('Procedure RefreshLanguage');
    RegisterMethod('Procedure ChangeWizardLanguage');
    RegisterMethod('Function WizardByName( const WizardName : string) : TCnBaseWizard');
    RegisterMethod('Function WizardByClass( AClass : TCnWizardClass) : TCnBaseWizard');
    RegisterMethod('Function WizardByClassName( const AClassName : string) : TCnBaseWizard');
    RegisterMethod('Function ImageIndexByClassName( const AClassName : string) : Integer');
    RegisterMethod('Function ActionByWizardClassNameAndCommand( const AClassName : string; const ACommand : string) : TCnWizAction');
    RegisterMethod('Function ImageIndexByWizardClassNameAndCommand( const AClassName : string; const ACommand : string) : Integer');
    RegisterMethod('Function IndexOf( Wizard : TCnBaseWizard) : Integer');
    RegisterMethod('Procedure DispatchDebugComand( Cmd : string; Results : TStrings)');
    RegisterProperty('Menu', 'TMenuItem', iptr);
    RegisterProperty('WizardCount', 'Integer', iptr);
    RegisterProperty('MenuWizardCount', 'Integer', iptr);
    RegisterProperty('IDEEnhanceWizardCount', 'Integer', iptr);
    RegisterProperty('RepositoryWizardCount', 'Integer', iptr);
    RegisterProperty('Wizards', 'TCnBaseWizard Integer', iptr);
    SetDefaultPropery('Wizards');
    RegisterProperty('MenuWizards', 'TCnMenuWizard Integer', iptr);
    RegisterProperty('IDEEnhanceWizards', 'TCnIDEEnhanceWizard Integer', iptr);
    RegisterProperty('RepositoryWizards', 'TCnRepositoryWizard Integer', iptr);
    RegisterProperty('WizardCanCreate', 'Boolean string', iptrw);
    RegisterProperty('OffSet', 'Integer Integer', iptr);
  end;
end;

(*----------------------------------------------------------------------------*)
procedure SIRegister_CnWizManager(CL: TPSPascalCompiler);
begin
 // CL.AddConstantN('BootShortCutKey','').SetString( VK_LSHIFT);
 CL.AddConstantN('CNWIZARDS_SETTING_WIZARDS_CHANGED','LongInt').SetInt( 1);
 CL.AddConstantN('CNWIZARDS_SETTING_PROPERTY_EDITORS_CHANGED','LongInt').SetInt( 2);
 CL.AddConstantN('CNWIZARDS_SETTING_COMPONENT_EDITORS_CHANGED','LongInt').SetInt( 4);
 CL.AddConstantN('CNWIZARDS_SETTING_OTHERS_CHANGED','LongInt').SetInt( 8);
 CL.AddConstantN('KEY_MAPPING_REG','String').SetString( '\Editor\Options\Known Editor Enhancements');
  SIRegister_TCnWizardMgr(CL);
  SIRegister_TCnDesignSelectionManager(CL);
 CL.AddDelphiFunction('Procedure RegisterBaseDesignMenuExecutor( Executor : TCnBaseMenuExecutor)');
 CL.AddDelphiFunction('Procedure RegisterDesignMenuExecutor( Executor : TCnContextMenuExecutor)');
 CL.AddDelphiFunction('Procedure UnRegisterBaseDesignMenuExecutor( Executor : TCnBaseMenuExecutor)');
 CL.AddDelphiFunction('Procedure UnRegisterDesignMenuExecutor( Executor : TCnContextMenuExecutor)');
 CL.AddDelphiFunction('Procedure RegisterEditorMenuExecutor( Executor : TCnContextMenuExecutor)');
 CL.AddDelphiFunction('Procedure UnRegisterEditorMenuExecutor( Executor : TCnContextMenuExecutor)');
 CL.AddDelphiFunction('Function GetEditorMenuExecutorCount : Integer');
 CL.AddDelphiFunction('Function GetEditorMenuExecutor( Index : Integer) : TCnContextMenuExecutor');
 CL.AddDelphiFunction('Function GetCnWizardMgr : TCnWizardMgr');
end;

(* === run-time registration functions === *)
(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrOffSet_R(Self: TCnWizardMgr; var T: Integer; const t1: Integer);
begin T := Self.OffSet[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrWizardCanCreate_W(Self: TCnWizardMgr; const T: Boolean; const t1: string);
begin Self.WizardCanCreate[t1] := T; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrWizardCanCreate_R(Self: TCnWizardMgr; var T: Boolean; const t1: string);
begin T := Self.WizardCanCreate[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrRepositoryWizards_R(Self: TCnWizardMgr; var T: TCnRepositoryWizard; const t1: Integer);
begin T := Self.RepositoryWizards[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrIDEEnhanceWizards_R(Self: TCnWizardMgr; var T: TCnIDEEnhanceWizard; const t1: Integer);
begin T := Self.IDEEnhanceWizards[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrMenuWizards_R(Self: TCnWizardMgr; var T: TCnMenuWizard; const t1: Integer);
begin T := Self.MenuWizards[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrWizards_R(Self: TCnWizardMgr; var T: TCnBaseWizard; const t1: Integer);
begin T := Self.Wizards[t1]; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrRepositoryWizardCount_R(Self: TCnWizardMgr; var T: Integer);
begin T := Self.RepositoryWizardCount; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrIDEEnhanceWizardCount_R(Self: TCnWizardMgr; var T: Integer);
begin T := Self.IDEEnhanceWizardCount; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrMenuWizardCount_R(Self: TCnWizardMgr; var T: Integer);
begin T := Self.MenuWizardCount; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrWizardCount_R(Self: TCnWizardMgr; var T: Integer);
begin T := Self.WizardCount; end;

(*----------------------------------------------------------------------------*)
procedure TCnWizardMgrMenu_R(Self: TCnWizardMgr; var T: TMenuItem);
begin T := Self.Menu; end;

(*----------------------------------------------------------------------------*)
procedure RIRegister_CnWizManager_Routines(S: TPSExec);
begin
 S.RegisterDelphiFunction(@RegisterBaseDesignMenuExecutor, 'RegisterBaseDesignMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@RegisterDesignMenuExecutor, 'RegisterDesignMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@UnRegisterBaseDesignMenuExecutor, 'UnRegisterBaseDesignMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@UnRegisterDesignMenuExecutor, 'UnRegisterDesignMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@RegisterEditorMenuExecutor, 'RegisterEditorMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@UnRegisterEditorMenuExecutor, 'UnRegisterEditorMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@GetEditorMenuExecutorCount, 'GetEditorMenuExecutorCount', cdRegister);
 S.RegisterDelphiFunction(@GetEditorMenuExecutor, 'GetEditorMenuExecutor', cdRegister);
 S.RegisterDelphiFunction(@GetCnWizardMgr, 'GetCnWizardMgr', cdRegister);
end;

{$IFDEF COMPILER6_UP}
(*----------------------------------------------------------------------------*)
procedure RIRegister_TCnDesignSelectionManager(CL: TPSRuntimeClassImporter);
begin
  with CL.Add(TCnDesignSelectionManager) do
  begin
    RegisterMethod(@TCnDesignSelectionManager.ExecuteVerb, 'ExecuteVerb');
    RegisterMethod(@TCnDesignSelectionManager.GetVerb, 'GetVerb');
    RegisterMethod(@TCnDesignSelectionManager.GetVerbCount, 'GetVerbCount');
    RegisterMethod(@TCnDesignSelectionManager.PrepareItem, 'PrepareItem');
    RegisterMethod(@TCnDesignSelectionManager.RequiresUnits, 'RequiresUnits');
  end;
end;
{$ENDIF}

(*----------------------------------------------------------------------------*)
procedure RIRegister_TCnWizardMgr(CL: TPSRuntimeClassImporter);
begin
  with CL.Add(TCnWizardMgr) do
  begin
    RegisterConstructor(@TCnWizardMgr.Create, 'Create');
    RegisterMethod(@TCnWizardMgr.GetIDString, 'GetIDString');
    RegisterMethod(@TCnWizardMgr.GetName, 'GetName');
    RegisterMethod(@TCnWizardMgr.GetState, 'GetState');
    RegisterMethod(@TCnWizardMgr.Execute, 'Execute');
    RegisterMethod(@TCnWizardMgr.LoadSettings, 'LoadSettings');
    RegisterMethod(@TCnWizardMgr.SaveSettings, 'SaveSettings');
    RegisterMethod(@TCnWizardMgr.ConstructSortedMenu, 'ConstructSortedMenu');
    RegisterMethod(@TCnWizardMgr.UpdateMenuPos, 'UpdateMenuPos');
    RegisterMethod(@TCnWizardMgr.RefreshLanguage, 'RefreshLanguage');
    RegisterMethod(@TCnWizardMgr.ChangeWizardLanguage, 'ChangeWizardLanguage');
    RegisterMethod(@TCnWizardMgr.WizardByName, 'WizardByName');
    RegisterMethod(@TCnWizardMgr.WizardByClass, 'WizardByClass');
    RegisterMethod(@TCnWizardMgr.WizardByClassName, 'WizardByClassName');
    RegisterMethod(@TCnWizardMgr.ImageIndexByClassName, 'ImageIndexByClassName');
    RegisterMethod(@TCnWizardMgr.ActionByWizardClassNameAndCommand, 'ActionByWizardClassNameAndCommand');
    RegisterMethod(@TCnWizardMgr.ImageIndexByWizardClassNameAndCommand, 'ImageIndexByWizardClassNameAndCommand');
    RegisterMethod(@TCnWizardMgr.IndexOf, 'IndexOf');
    RegisterMethod(@TCnWizardMgr.DispatchDebugComand, 'DispatchDebugComand');
    RegisterPropertyHelper(@TCnWizardMgrMenu_R,nil,'Menu');
    RegisterPropertyHelper(@TCnWizardMgrWizardCount_R,nil,'WizardCount');
    RegisterPropertyHelper(@TCnWizardMgrMenuWizardCount_R,nil,'MenuWizardCount');
    RegisterPropertyHelper(@TCnWizardMgrIDEEnhanceWizardCount_R,nil,'IDEEnhanceWizardCount');
    RegisterPropertyHelper(@TCnWizardMgrRepositoryWizardCount_R,nil,'RepositoryWizardCount');
    RegisterPropertyHelper(@TCnWizardMgrWizards_R,nil,'Wizards');
    RegisterPropertyHelper(@TCnWizardMgrMenuWizards_R,nil,'MenuWizards');
    RegisterPropertyHelper(@TCnWizardMgrIDEEnhanceWizards_R,nil,'IDEEnhanceWizards');
    RegisterPropertyHelper(@TCnWizardMgrRepositoryWizards_R,nil,'RepositoryWizards');
    RegisterPropertyHelper(@TCnWizardMgrWizardCanCreate_R,@TCnWizardMgrWizardCanCreate_W,'WizardCanCreate');
    RegisterPropertyHelper(@TCnWizardMgrOffSet_R,nil,'OffSet');
  end;
end;

(*----------------------------------------------------------------------------*)
procedure RIRegister_CnWizManager(CL: TPSRuntimeClassImporter);
begin
  RIRegister_TCnWizardMgr(CL);
{$IFDEF COMPILER6_UP}
  RIRegister_TCnDesignSelectionManager(CL);
{$ENDIF}
end;



{ TPSImport_CnWizManager }
(*----------------------------------------------------------------------------*)
procedure TPSImport_CnWizManager.CompileImport1(CompExec: TPSScript);
begin
  SIRegister_CnWizManager(CompExec.Comp);
end;
(*----------------------------------------------------------------------------*)
procedure TPSImport_CnWizManager.ExecImport1(CompExec: TPSScript; const ri: TPSRuntimeClassImporter);
begin
  RIRegister_CnWizManager(ri);
  RIRegister_CnWizManager_Routines(CompExec.Exec); // comment it if no routines
end;
(*----------------------------------------------------------------------------*)


end.
