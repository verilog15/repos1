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

unit CnWizManager;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�CnWizardMgr ר�ҹ�����ʵ�ֵ�Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע���õ�ԪΪ CnWizards ��ܵ�һ���֣������� CnWizardMgr ר�ҹ�������
*           ��Ԫʵ����ר�� DLL ����ڵ�������������ר�ҹ�����ʼ�����е�ר�ҡ�
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2015.05.19 V1.3 by liuxiao
*               ���� D6 ���ϰ汾��ע��������Ҽ��˵�ִ����Ļ���
*           2003.10.03 V1.2 by ����(QSoft)
*               ����ר����������
*           2003.08.02 V1.1
*               LiuXiao ���� WizardCanCreate ���ԡ�
*           2002.09.17 V1.0
*               ������Ԫ��ʵ�ֻ�������
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF IDE_INTEGRATE_CASTALIA}
  {$IFNDEF DELPHI101_BERLIN_UP}
    {$DEFINE CASTALIA_KEYMAPPING_CONFLICT_BUG}
  {$ENDIF}
{$ENDIF}

uses
  Windows, Messages, Classes, Graphics, Controls, Sysutils, Menus, ActnList,
  Forms, ImgList, ExtCtrls, IniFiles, Dialogs, Registry, ToolsAPI, Contnrs,
  {$IFDEF COMPILER6_UP}
  DesignIntf, DesignEditors, DesignMenus,
  {$ELSE}
  DsgnIntf,
  {$ENDIF}
  CnWizClasses, CnWizConsts, CnWizMenuAction
  {$IFNDEF CNWIZARDS_MINIMUM}, CnLangMgr, CnRestoreSystemMenu, CnWizIdeHooks {$ENDIF};

const
  BootShortCutKey = VK_LSHIFT; // ��ݼ�Ϊ �� Shift���û����������� Delphi ʱ
                               // ���¸ü�������ר����������

  // ���øı�ʱ֪ͨ��ȥ��֪ͨ����õĴ��Եĸı����ݣ�
  // ����ר��ʹ�ܡ�����/����༭��ʹ�ܡ�������
  CNWIZARDS_SETTING_WIZARDS_CHANGED            = 1;
  CNWIZARDS_SETTING_PROPERTY_EDITORS_CHANGED   = 2;
  CNWIZARDS_SETTING_COMPONENT_EDITORS_CHANGED  = 4;
  CNWIZARDS_SETTING_OTHERS_CHANGED             = 8;

const
  KEY_MAPPING_REG = '\Editor\Options\Known Editor Enhancements';

type

//==============================================================================
// TCnWizardMgr ��ר����
//==============================================================================

{ TCnWizardMgr }

  TCnWizardMgr = class(TNotifierObject, IOTAWizard)
  {* CnWizardMgr ר�ҹ������࣬����ά��ר���б�
     �벻Ҫֱ�Ӵ��������ʵ���������ʵ����ר�� DLL ע��ʱ�Զ���������ʹ��ȫ��
     ���� CnWizardMgr �����ʹ�����ʵ����}
  private
{$IFNDEF CNWIZARDS_MINIMUM}
    FRestoreSysMenu: TCnRestoreSystemMenu;
{$ENDIF}
    FMenu: TMenuItem;
    FToolsMenu: TMenuItem;
    FWizards: TList;
    FMenuWizards: TList;
    FIDEEnhanceWizards: TList;
    FRepositoryWizards: TList;
    FTipTimer: TTimer;
    FLaterLoadTimer: TTimer;
    FSepMenu: TMenuItem;
    FConfigAction: TCnWizMenuAction;
    FWizMultiLang: TCnMenuWizard;
    FWizAbout: TCnMenuWizard;
    FOffSet: array[0..3] of Integer;
    FSettingsLoaded: Boolean;
  {$IFDEF BDS}
    FSplashBmp: TBitmap;
    FAboutBmp: TBitmap;
  {$ENDIF}
    procedure DoLaterLoad(Sender: TObject);
    procedure DoFreeLaterLoadTimer(Sender: TObject);

    procedure CreateIDEMenu;
    procedure InstallIDEMenu;
    procedure FreeMenu;
    procedure InstallWizards;
    procedure FreeWizards;
    procedure CreateMiscMenu;
    procedure InstallMiscMenu;
    procedure EnsureNoParent(Menu: TMenuItem);
    procedure FreeMiscMenu;

    procedure RegisterPluginInfo;
    procedure InternalCreate;
    procedure InstallPropEditors;
    procedure InstallCompEditors;
    procedure SetTipShowing;
    procedure ShowTipofDay(Sender: TObject);
    procedure CheckIDEVersion;
{$IFNDEF CNWIZARDS_MINIMUM}
{$IFDEF CASTALIA_KEYMAPPING_CONFLICT_BUG}
    procedure CheckKeyMappingEnhModulesSequence;
{$ENDIF}
{$ENDIF}
    function GetWizards(Index: Integer): TCnBaseWizard;
    function GetWizardCount: Integer;
    function GetMenuWizardCount: Integer;
    function GetMenuWizards(Index: Integer): TCnMenuWizard;
    function GetRepositoryWizardCount: Integer;
    function GetRepositoryWizards(Index: Integer): TCnRepositoryWizard;
    procedure OnConfig(Sender: TObject);
    procedure OnIdleLoaded(Sender: TObject);
    procedure OnFileNotify(NotifyCode: TOTAFileNotification; const FileName: string);
    function GetIDEEnhanceWizardCount: Integer;
    function GetIDEEnhanceWizards(Index: Integer): TCnIDEEnhanceWizard;
    function GetWizardCanCreate(WizardClassName: string): Boolean;
    procedure SetWizardCanCreate(WizardClassName: string;
      const Value: Boolean);
    function GetOffSet(Index: Integer): Integer;
  public
    constructor Create;
    {* �๹����}
    destructor Destroy; override;
    {* ��������}

    // IOTAWizard methods
    function GetIDString: string;
    function GetName: string;
    function GetState: TWizardState;
    procedure Execute;

    procedure LoadSettings;
    {* װ������ר�ҵ�����}
    procedure SaveSettings;
    {* ��������ר�ҵ�����}
    procedure ConstructSortedMenu;
    {* �ؽ������Ĳ˵� }
    procedure UpdateMenuPos(UseToolsMenu: Boolean);
    {* �����������˵���λ�ã��ж������˵������� Tools �� }
    procedure RefreshLanguage;
    {* ���¶���ר�ҵĸ��������ַ��������� Action ���� }
    procedure ChangeWizardLanguage;
    {* ����ר�ҵ����Ըı��¼�������ר�ҵ����� }
    function WizardByName(const WizardName: string): TCnBaseWizard;
    {* ����ר�����Ʒ���ר��ʵ��������Ҳ���ר�ң�����Ϊ nil}
    function WizardByClass(AClass: TCnWizardClass): TCnBaseWizard;
    {* ����ר����������ר��ʵ��������Ҳ���ר�ң�����Ϊ nil}
    function WizardByClassName(const AClassName: string): TCnBaseWizard;
    {* ����ר�������ַ�������ר��ʵ��������Ҳ���ר�ң�����Ϊ nil}
    function ImageIndexByClassName(const AClassName: string): Integer;
    {* ����ר�������ַ�������ר�ҵ�ͼ������������Ҳ���ר�һ���ͼ������������Ϊ -1}
    function ActionByWizardClassNameAndCommand(const AClassName: string;
      const ACommand: string): TCnWizAction;
    {* ����ר�������ַ����������ַ��ظ��Ӳ˵�ר�ҵ�ָ���� Action�����򷵻� nil}
    function ImageIndexByWizardClassNameAndCommand(const AClassName: string;
      const ACommand: string): Integer;
    {* ����ר�������ַ����������ַ��ظ��Ӳ˵�ר�ҵ�ָ���� Action �� ImageIndex�����򷵻� -1}
    function IndexOf(Wizard: TCnBaseWizard): Integer;
    {* ����ר��ʵ����������ר���б��е�������}
    procedure DispatchDebugComand(Cmd: string; Results: TStrings);
    {* �ַ����� Debug ����������������� Results �У����ڲ�������}
    property Menu: TMenuItem read FMenu;
    {* ���뵽 IDE ���˵��еĲ˵���}
    property WizardCount: Integer read GetWizardCount;
    {* TCnBaseWizard ��������ר�ҵ����������������е�ר��}
    property MenuWizardCount: Integer read GetMenuWizardCount;
    {* TCnMenuWizard �˵�ר�Ҽ������������}
    property IDEEnhanceWizardCount: Integer read GetIDEEnhanceWizardCount;
    {* TCnIDEEnhanceWizard ר�Ҽ������������}
    property RepositoryWizardCount: Integer read GetRepositoryWizardCount;
    {* TCnRepositoryWizard ģ����ר�Ҽ������������}
    property Wizards[Index: Integer]: TCnBaseWizard read GetWizards; default;
    {* ר�����飬�����˹�����ά��������ר��}
    property MenuWizards[Index: Integer]: TCnMenuWizard read GetMenuWizards;
    {* �˵�ר�����飬������ TCnMenuWizard ��������ר��}
    property IDEEnhanceWizards[Index: Integer]: TCnIDEEnhanceWizard
      read GetIDEEnhanceWizards;
    {* IDE ������չר�����飬������ TCnIDEEnhanceWizard ��������ר��}
    property RepositoryWizards[Index: Integer]: TCnRepositoryWizard
      read GetRepositoryWizards;
    {* ģ����ר�����飬������ TCnRepositoryWizard ��������ר��}

    property WizardCanCreate[WizardClassName: string]: Boolean read GetWizardCanCreate
      write SetWizardCanCreate;
    {* ָ��ר���Ƿ񴴽� }
    property OffSet[Index: Integer]: Integer read GetOffSet;
  end;

{$IFDEF COMPILER6_UP}

  TCnDesignSelectionManager = class(TBaseSelectionEditor, ISelectionEditor)
  {* ������Ҽ��˵�ִ����Ŀ������}
  public
    procedure ExecuteVerb(Index: Integer; const List: IDesignerSelections);
    function GetVerb(Index: Integer): string;
    function GetVerbCount: Integer;
    procedure PrepareItem(Index: Integer; const AItem: IMenuItem);
    procedure RequiresUnits(Proc: TGetStrProc);
  end;

{$ENDIF}

var
  CnWizardMgr: TCnWizardMgr = nil;
  {* TCnWizardMgr ��ר��ʵ��}

  InitSplashProc: TProcedure = nil;
  {* ������洰��ͼƬ�����ݵ����ģ��}

procedure RegisterBaseDesignMenuExecutor(Executor: TCnBaseMenuExecutor);
{* ע��һ��������Ҽ��˵���ִ�ж���ʵ����Ӧ����ר�Ҵ���ʱע��
  ע��˷������ú�Executor ���ɴ˴�ͳһ���������ͷţ������ⲿ�����ͷ���}

procedure RegisterDesignMenuExecutor(Executor: TCnContextMenuExecutor);
{* ע��һ��������Ҽ��˵���ִ�ж���ʵ������һ��ʽ}

procedure UnRegisterBaseDesignMenuExecutor(Executor: TCnBaseMenuExecutor);
{* ��ע��һ��������Ҽ��˵���ִ�ж���ʵ������ע��� Executor ���Զ��ͷ�}

procedure UnRegisterDesignMenuExecutor(Executor: TCnContextMenuExecutor);
{* ��ע��һ��������Ҽ��˵���ִ�ж���ʵ������һ��ʽ����ע��� Executor ���Զ��ͷ�}

procedure RegisterEditorMenuExecutor(Executor: TCnContextMenuExecutor);
{* ע��һ���༭���Ҽ��˵���ִ�ж���ʵ����Ӧ����ר�Ҵ���ʱע��}

procedure UnRegisterEditorMenuExecutor(Executor: TCnContextMenuExecutor);
{* ��ע��һ���༭���Ҽ��˵���ִ�ж���ʵ������ע��� Executor ���Զ��ͷ�}

function GetEditorMenuExecutorCount: Integer;
{* ������ע��ı༭���Ҽ��˵���Ŀ���������༭����չʵ���Զ���༭���˵���}

function GetEditorMenuExecutor(Index: Integer): TCnContextMenuExecutor;
{* ������ע��ı༭���Ҽ��˵���Ŀ�����༭����չʵ���Զ���༭���˵���}

function GetCnWizardMgr: TCnWizardMgr;
{* ��װ�ķ��� TCnWizardMgr ��ר��ʵ���ĺ�������Ҫ���ű�ר��ʹ��}

implementation

uses
{$IFDEF DEBUG}
  CnDebug, 
{$ENDIF}
  CnWizUtils, CnWizOptions, CnWizShortCut, CnCommon,
{$IFNDEF CNWIZARDS_MINIMUM}
  CnWizConfigFrm, CnWizAbout, CnWizShareImages,
  CnWizUpgradeFrm, CnDesignEditor, CnWizMultiLang, CnWizBoot,
  CnWizCommentFrm, CnWizTranslate, CnWizTipOfDayFrm, CnIDEVersion,
{$ENDIF}
  CnWizNotifier, CnWizCompilerConst;

function GetCnWizardMgr: TCnWizardMgr;
begin
end;

procedure RegisterBaseDesignMenuExecutor(Executor: TCnBaseMenuExecutor);
begin
end;

procedure RegisterDesignMenuExecutor(Executor: TCnContextMenuExecutor);
begin
end;

procedure UnRegisterBaseDesignMenuExecutor(Executor: TCnBaseMenuExecutor);
begin
end;

procedure UnRegisterDesignMenuExecutor(Executor: TCnContextMenuExecutor);
begin
end;

procedure RegisterEditorMenuExecutor(Executor: TCnContextMenuExecutor);
begin
end;

procedure UnRegisterEditorMenuExecutor(Executor: TCnContextMenuExecutor);
begin
end;

function GetEditorMenuExecutorCount: Integer;
begin
end;

function GetEditorMenuExecutor(Index: Integer): TCnContextMenuExecutor;
begin
end;

procedure TCnWizardMgr.InternalCreate;
begin
end;

procedure TCnWizardMgr.RegisterPluginInfo;
begin
end;

procedure TCnWizardMgr.DoFreeLaterLoadTimer(Sender: TObject);
begin
end;

procedure TCnWizardMgr.DoLaterLoad(Sender: TObject);
begin
end;

constructor TCnWizardMgr.Create;
begin
end;

destructor TCnWizardMgr.Destroy;
begin
end;

procedure TCnWizardMgr.CreateIDEMenu;
begin
end;

procedure TCnWizardMgr.InstallIDEMenu;
begin
end;

procedure TCnWizardMgr.RefreshLanguage;
begin
end;

procedure TCnWizardMgr.ChangeWizardLanguage;
begin
end;

procedure TCnWizardMgr.CreateMiscMenu;
begin
end;

procedure TCnWizardMgr.ConstructSortedMenu;
begin
end;

procedure TCnWizardMgr.UpdateMenuPos(UseToolsMenu: Boolean);
begin
end;

function TCnWizardMgr.GetIDEEnhanceWizardCount: Integer;
begin
end;

function TCnWizardMgr.GetIDEEnhanceWizards(Index: Integer): TCnIDEEnhanceWizard;
begin
end;

function TCnWizardMgr.WizardByClass(AClass: TCnWizardClass): TCnBaseWizard;
begin
end;

function TCnWizardMgr.WizardByClassName(const AClassName: string): TCnBaseWizard;
begin
end;

function TCnWizardMgr.ImageIndexByClassName(const AClassName: string): Integer;
begin
end;

function TCnWizardMgr.ActionByWizardClassNameAndCommand(const AClassName: string;
  const ACommand: string): TCnWizAction;
begin
end;

function TCnWizardMgr.ImageIndexByWizardClassNameAndCommand(const AClassName: string;
  const ACommand: string): Integer;
begin
end;

function TCnWizardMgr.IndexOf(Wizard: TCnBaseWizard): Integer;
begin
end;

function TCnWizardMgr.WizardByName(const WizardName: string): TCnBaseWizard;
begin
end;

procedure TCnWizardMgr.FreeMenu;
begin
end;

procedure TCnWizardMgr.InstallWizards;
begin
end;

function TCnWizardMgr.GetOffSet(Index: Integer): Integer;
begin
end;

procedure TCnWizardMgr.FreeWizards;
begin
end;

procedure TCnWizardMgr.LoadSettings;
begin
end;

procedure TCnWizardMgr.SaveSettings;
begin
end;

procedure TCnWizardMgr.EnsureNoParent(Menu: TMenuItem);
begin
end;

procedure TCnWizardMgr.InstallMiscMenu;
begin
end;

procedure TCnWizardMgr.FreeMiscMenu;
begin
end;

procedure TCnWizardMgr.InstallCompEditors;
begin
end;

procedure TCnWizardMgr.InstallPropEditors;
begin
end;

procedure TCnWizardMgr.SetTipShowing;
begin
end;

procedure TCnWizardMgr.ShowTipofDay(Sender: TObject);
begin
end;

procedure TCnWizardMgr.CheckIDEVersion;
begin
end;

procedure TCnWizardMgr.OnFileNotify(NotifyCode: TOTAFileNotification;
  const FileName: string);
begin
end;

procedure TCnWizardMgr.OnIdleLoaded(Sender: TObject);
begin
end;

procedure TCnWizardMgr.OnConfig(Sender: TObject);
begin
end;

function TCnWizardMgr.GetWizardCount: Integer;
begin
end;

function TCnWizardMgr.GetWizards(Index: Integer): TCnBaseWizard;
begin
end;

function TCnWizardMgr.GetMenuWizardCount: Integer;
begin
end;

function TCnWizardMgr.GetMenuWizards(Index: Integer): TCnMenuWizard;
begin
end;

function TCnWizardMgr.GetRepositoryWizardCount: Integer;
begin
end;

function TCnWizardMgr.GetRepositoryWizards(
  Index: Integer): TCnRepositoryWizard;
begin
end;

function TCnWizardMgr.GetWizardCanCreate(WizardClassName: string): Boolean;
begin
end;

procedure TCnWizardMgr.SetWizardCanCreate(WizardClassName: string;
  const Value: Boolean);
begin
end;

procedure TCnWizardMgr.DispatchDebugComand(Cmd: string; Results: TStrings);
begin
end;

procedure TCnWizardMgr.Execute;
begin
end;

function TCnWizardMgr.GetIDString: string;
begin
end;

function TCnWizardMgr.GetName: string;
begin
end;

function TCnWizardMgr.GetState: TWizardState;
begin
end;

procedure TCnWizardMgr.CheckKeyMappingEnhModulesSequence;
begin
end;

procedure TCnDesignSelectionManager.ExecuteVerb(Index: Integer;
  const List: IDesignerSelections);
begin
end;

function TCnDesignSelectionManager.GetVerb(Index: Integer): string;
begin
end;

function TCnDesignSelectionManager.GetVerbCount: Integer;
begin
end;

procedure TCnDesignSelectionManager.PrepareItem(Index: Integer;
  const AItem: IMenuItem);
begin
end;

procedure TCnDesignSelectionManager.RequiresUnits(Proc: TGetStrProc);
begin
end;

end.
