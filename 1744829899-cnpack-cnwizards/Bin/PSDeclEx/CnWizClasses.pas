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

unit CnWizClasses;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�CnWizards �����ඨ�嵥Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע���õ�ԪΪ CnWizards ��ܵ�һ���֣�����������ר�ҵĻ����ࡣ
*           Ҫע��һ����ʵ�ֵ�ר�ң�����ʵ�ָ�ר�ҵĵ�Ԫ initialization �ڵ���
*           RegisterCnWizard ��ע��һ��ר�������á�
*         - TCnBaseWizard
*           ���� CnWizard ��ײ�ĳ�����ࡣ
*           - TCnIconWizard
*             ��ͼ��ĳ�����ࡣ
*             - TCnIDEEnhanceWizard
*               IDE ������չר�һ��ࡣ
*             - TCnActionWizard
*               �� IDE Action �ĳ�����࣬�������������п�ݼ���ר�ҡ�
*               - TCnMenuWizard
*                 ���˵��ĵĳ�����࣬������������ͨ���˵����õ�ר�ҡ�
*                 - TCnSubMenuWizard
*                   ���Ӳ˵���ĳ�����࣬������������ͨ���Ӳ˵�����õ�ר�ҡ�
*             - TCnRepositoryWizard
*               ���� Repository ר�һ��ࡣ
*               - TCnFormWizard
*                 �����������嵥Ԫ�ļ���ģ���򵼻��࣬���������� Pas ��Ԫ��
*               - TCnProjectWizard
*                 ��������Ӧ�ó��򹤳̵�ģ���򵼻��ࡣ
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2002.09.17 V1.0
*               ������Ԫ��ʵ�ֻ�������
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Classes, Sysutils, Graphics, Menus, ActnList, IniFiles, ToolsAPI,
  Registry, ComCtrls, Forms, CnHashMap, CnWizIni,
  CnWizShortCut, CnWizMenuAction, CnIni, CnWizConsts, CnPopupMenu;

type

//==============================================================================
// ר�Ұ�ר�ҳ������
//==============================================================================

{ TCnBaseWizard }

{$M+}

  TCnBaseWizard = class(TNotifierObject, IOTAWizard)
  {* CnWizard ר�ҳ�����࣬������ר��������Ĺ������� }
  private
    FActive: Boolean;
    FWizardIndex: Integer;
    FDefaultsMap: TCnStrToVariantHashMap;
  protected
    procedure SetActive(Value: Boolean); virtual;
    {* Active ����д�������������ظ÷������� Active ���Ա���¼� }
    function GetHasConfig: Boolean; virtual;
    {* HasConfig ���Զ��������������ظ÷��������Ƿ���ڿ��������� }
    function GetIcon: TIcon; virtual; abstract;
    {* Icon ���Զ��������������ظ÷������ط���ר��ͼ�꣬�û�ר��ͨ�������Լ����� }
    function GetBigIcon: TIcon; virtual; abstract;
    {* ��ߴ� Icon ���Զ��������������ظ÷������ط���ר��ͼ�꣬�û�ר��ͨ�������Լ����� }

    // IOTAWizard methods
    function GetIDString: string;
    function GetName: string; virtual;
  public
    constructor Create; virtual;
    {* �๹���� }
    destructor Destroy; override;
    {* �������� }
    class function WizardName: string;
    {* ȡר�����ƣ�������֧�ֱ��ػ����ַ��� }
    function GetAuthor: string; virtual;
    {* ��������}
    function GetComment: string; virtual;
    {* ����ע��}
    function GetSearchContent: string; virtual;
    {* ���ع��������ַ������������԰�Ƕ��ŷָ����Ӣ�Ĺؼ��ʣ���Ҫ��Сд}
    procedure DebugComand(Cmds: TStrings; Results: TStrings); virtual;
    {* ���� Debug ����������������� Results �У����ڲ�������}

    // IOTAWizard methods
    function GetState: TWizardState; virtual;
    {* ����ר��״̬��IOTAWizard �������������ظ÷�������ר��״̬ }
    procedure Execute; virtual; abstract;
    {* ר��ִ���巽����IOTAWizard ���󷽷����������ʵ�֡�
       ���û�ִ��һ��ר��ʱ�������ø÷����� }

    procedure Loaded; virtual;
    {* IDE ������ɺ���ø÷���}

    procedure LaterLoaded; virtual;
    {* IDE ������ɸ���һЩ����ø÷��������ڸ߰汾 IDE �д��� IDE �˵������̫�ٵĳ���}

    class function IsInternalWizard: Boolean; virtual;
    {* ��ר���Ƿ������ڲ�ר�ң�����ʾ���������� }

    class procedure GetWizardInfo(var Name, Author, Email, Comment: string);
      virtual; {$IFNDEF BCB}abstract;{$ENDIF BCB}
    {* ȡר����Ϣ�������ṩר�ҵ�˵���Ͱ�Ȩ��Ϣ�����󷽷����������ʵ�֡�
     |<PRE>
       var AName: string      - ר�����ƣ�������֧�ֱ��ػ����ַ���
       var Author: string     - ר�����ߣ�����ж�����ߣ��÷ֺŷָ�
       var Email: string      - ר���������䣬����ж�����ߣ��÷ֺŷָ�
       var Comment: string    - ר��˵����������֧�ֱ��ػ������з����ַ���
     |</PRE> }
    procedure Config; virtual;
    {* ר�����÷�������ר�ҹ�������ר�����ý����е��ã��� HasConfig Ϊ��ʱ��Ч }
    procedure LanguageChanged(Sender: TObject); virtual;
    {* ר���ڶ�������Է����ı��ʱ��Ĵ����������ش˹��̴��������ַ��� }
    procedure LoadSettings(Ini: TCustomIniFile); virtual;
    {* װ��ר�����÷������������ش˷����� INI �����ж�ȡר�Ҳ�����
       ע��˷���������ר��һ�����������б����ö�Σ������Ҫ��������
       ��ֹ�ظ����ص��¶������ݵ����⡣ }
    procedure SaveSettings(Ini: TCustomIniFile); virtual;
    {* ����ר�����÷������������ش˷�����ר�Ҳ������浽 INI ������ }
    procedure ResetSettings(Ini: TCustomIniFile); virtual;
    {* ����ר�����÷������������� INI ����֮��ı��涯������Ҫ���ش˷��� }

    class function GetIDStr: string;
    {* ����ר��Ψһ��ʶ������������ʹ�� }
    function CreateIniFile(CompilerSection: Boolean = False): TCustomIniFile;
    {* ����һ�����ڴ�ȡר�����ò����� INI �����û�ʹ�ú����Լ��ͷ� }
    procedure DoLoadSettings;
    {* װ��ר������ }
    procedure DoSaveSettings;
    {* ����ר������ }
    procedure DoResetSettings;
    {* ����ר������}

    property Active: Boolean read FActive write SetActive;
    {* ��Ծ���ԣ�����ר�ҵ�ǰ�Ƿ���� }
    property HasConfig: Boolean read GetHasConfig;
    {* ��ʾר���Ƿ�������ý�������� }
    property WizardIndex: Integer read FWizardIndex write FWizardIndex;
    {* ר��ע����� IDE ���ص������ţ����ͷ�ר��ʱʹ�ã��벻Ҫ�޸ĸ�ֵ }
    property Icon: TIcon read GetIcon;
    {* ר��ͼ�����ԣ���С��������������� ActionWizard �������� 16x16���� IconWizard Ĭ�� 32x32 }
    property BigIcon: TIcon read GetBigIcon;
    {* ר�Ҵ�ͼ�꣬����еĻ�}
  end;

{$M-}

type
  TCnWizardClass = class of TCnBaseWizard;

//==============================================================================
// ��ͼ�����ԵĻ�����
//==============================================================================

{ TCnIconWizard }

  TCnIconWizard = class(TCnBaseWizard)
  {* IDE ��ͼ�����ԵĻ����� }
  private
    FIcon: TIcon;
  protected
    function GetIcon: TIcon; override;
    {* ���ظ���ר�ҵ�ͼ�꣬��������ش˹��̷��������� Icon ����
       FIcon ʹ��ϵͳĬ�ϳߴ磬һ���� 32 * 32 ��}
    function GetBigIcon: TIcon; override;
    {* ���ظ���ר�ҵĴ�ͼ�꣬32 * 32 ��}
    procedure InitIcon(AIcon, ASmallIcon: TIcon); virtual;
    {* ����������ʼ��ͼ�꣬���󴴽�ʱ���ã���������ش˹������´��� FIcon }
    class function GetIconName: string; virtual;
    {* ����ͼ���ļ��� }
  public
    constructor Create; override;
    {* �๹���� }
    destructor Destroy; override;
    {* �������� }
  end;

//==============================================================================
// IDE ������չ������
//==============================================================================

{ TCnIDEEnhanceWizard }

  TCnIDEEnhanceWizard = class(TCnIconWizard);
  {* IDE ������չ������ }

//==============================================================================
// �� Action �Ϳ�ݼ��ĳ���ר����
//==============================================================================

{ TCnActionWizard }

  TCnActionWizard = class(TCnIDEEnhanceWizard)
  {* �� Action �Ϳ�ݼ��� CnWizard ר�ҳ�����࣬�Ӹ�����Icon �� 16x16 }
  private
    FAction: TCnWizAction;
    function GetImageIndex: Integer;
  protected
    function GetIcon: TIcon; override;
    procedure OnActionUpdate(Sender: TObject); virtual;
    function CreateAction: TCnWizAction; virtual;
    procedure Click(Sender: TObject); virtual;
    function GetCaption: string; virtual; abstract;
    {* ����ר�ҵı��� }
    function GetHint: string; virtual;
    {* ����ר�ҵ� Hint ��ʾ }
    function GetDefShortCut: TShortCut; virtual;
    {* ����ר�ҵ�Ĭ�Ͽ�ݼ���ʵ��ʹ��ʱר�ҵĿ�ݼ�������ɹ��������趨������
       ֻ��Ҫ����Ĭ�ϵľ����ˡ� }
  public
    constructor Create; override;
    {* �๹���� }
    destructor Destroy; override;
    {* �������� }
    function GetSearchContent: string; override;
    {* ���ع��������ַ������ѱ����� Hint ����ȥ }
    property ImageIndex: Integer read GetImageIndex;
    {* ר��ͼ���� IDE ���� ImageList �е������� }
    property Action: TCnWizAction read FAction;
    {* ר�� Action ���� }
    function EnableShortCut: Boolean; virtual;
    {* ����ר���Ƿ�����ÿ�ݼ����� }
    procedure RefreshAction; virtual;
    {* ���¸��� Action ������ }
  end;

//==============================================================================
// ���˵��ĳ���ר����
//==============================================================================

{ TCnMenuWizard }

type
  TCnMenuWizard = class(TCnActionWizard)
  {* ���˵��� CnWizard ר�ҳ������ }
  private
    FMenuOrder: Integer;
    function GetAction: TCnWizMenuAction;
    function GetMenu: TMenuItem;
    procedure SetMenuOrder(const Value: Integer);
  protected
    function CreateAction: TCnWizAction; override;
  public
    constructor Create; override;

    function EnableShortCut: Boolean; override;
    {* ����ר���Ƿ�����ÿ�ݼ����� }
    property Menu: TMenuItem read GetMenu;
    {* ר�ҵĲ˵����� }
    property Action: TCnWizMenuAction read GetAction;
    {* ר�� Action ���� }
    property MenuOrder: Integer read FMenuOrder write SetMenuOrder;
  end;

//==============================================================================
// ���Ӳ˵���ĳ���ר����
//==============================================================================

{ TCnSubMenuWizard }

  TCnSubMenuWizard = class(TCnMenuWizard)
  {* ���Ӳ˵���� CnWizard ר�ҳ������ }
  private
    FList: TList;
    FPopupMenu: TPopupMenu;
    FPopupAction: TCnWizAction; // ���ڷ��õ���������ʱ�����İ�ť��Ӧ�� Action������ Action ͼ���ظ�
    FExecuting: Boolean;
    FRefreshing: Boolean;
    procedure FreeSubMenus;
    procedure OnExecute(Sender: TObject);
    procedure OnUpdate(Sender: TObject);
    procedure OnPopup(Sender: TObject);
    function GetSubActions(Index: Integer): TCnWizMenuAction;
    function GetSubActionCount: Integer;
    function GetSubMenus(Index: Integer): TMenuItem;
  protected
    procedure SetActive(Value: Boolean); override;
    procedure OnActionUpdate(Sender: TObject); override;
    function CreateAction: TCnWizAction; override;
    procedure Click(Sender: TObject); override;
    function IndexOf(SubAction: TCnWizMenuAction): Integer;
    {* ����ָ���� Action ���б��е������� }
    function RegisterASubAction(const ACommand, ACaption: string;
      AShortCut: TShortCut = 0; const AHint: string = '';
      const AIconName: string = ''): Integer;
    {* ע��һ���� Action�����������š�
     |<PRE>
       ACommand: string         - Action �����֣�����Ϊһ��Ψһ���ַ���ֵ
       ACaption: string         - Action �ı���
       AShortCut: TShortCut     - Action ��Ĭ�Ͽ�ݼ���ʵ��ʹ�õļ�ֵ���ע����ж�ȡ
       AHint: string            - Action ����ʾ��Ϣ
       Result: Integer          - �����б��е�������
     |</PRE> }
    procedure AddSubMenuWizard(SubMenuWiz: TCnSubMenuWizard);
    {* Ϊ��ר�ҹҽ�һ���Ӳ˵�ר�� }
    procedure AddSepMenu;
    {* ����һ���ָ��˵� }
    procedure DeleteSubAction(Index: Integer);
    {* ɾ��ָ������ Action }

    function ShowShortCutDialog(const HelpStr: string): Boolean;
    {* ��ʾ�� Action ��ݼ����öԻ��� }

    procedure SubActionExecute(Index: Integer); virtual;
    {* �Ӳ˵���ִ�з���������Ϊ�Ӳ˵��������ţ�ר�������ظ÷������� Action ִ���¼� }
    procedure SubActionUpdate(Index: Integer); virtual;
    {* �Ӳ˵�����·���������Ϊ�Ӳ˵��������ţ�ר�������ظ÷������� Action ״̬ }
  public
    constructor Create; override;
    {* �๹���� }
    destructor Destroy; override;
    {* �������� }
    function GetSearchContent: string; override;
    {* ���ع��������ַ������������Ӳ˵��ı����� Hint ����ȥ }
    procedure DebugComand(Cmds: TStrings; Results: TStrings); override;
    {* ����ʱ��ӡ�Ӳ˵��Լ� Action �ȵ���Ϣ}
    procedure Execute; override;
    {* ִ����ʲô������ }
    function EnableShortCut: Boolean; override;
    {* �����Ƿ�����ÿ�ݼ����� False }
    procedure AcquireSubActions; virtual;
    {* �������ش˹��̣��ڲ����� RegisterASubAction �����Ӳ˵��
        �˹����ڶ����л�ʱ�ᱻ�ظ����á� }
    procedure ClearSubActions; virtual;
    {* ɾ�����е��� Action�������Ӳ˵��еķָ��� }
    procedure RefreshAction; override;
    {* ���ص�ˢ�� Action �ķ��������˼̳�ˢ�²˵����⣬��ˢ���Ӳ˵� Action }
    procedure RefreshSubActions; virtual;
    {* ���������� Action������������ش˷�������ֹ�� Action ���� }
    property SubActionCount: Integer read GetSubActionCount;
    {* ר���� Action ���� }
    property SubMenus[Index: Integer]: TMenuItem read GetSubMenus;
    {* ר�ҵ��Ӳ˵��������� }
    property SubActions[Index: Integer]: TCnWizMenuAction read GetSubActions;
    {* ר�ҵ��� Action �������� }
    function ActionByCommand(const ACommand: string): TCnWizAction;
    {* ����ָ�������ֲ����� Action�����򷵻� nil}
  end;

//==============================================================================
// ���� Repository ר�һ���
//==============================================================================

{ TCnRepositoryWizard }

  TCnRepositoryWizard = class(TCnIconWizard, IOTARepositoryWizard)
  {* CnWizard ģ���򵼳������ }
  protected
    FIconHandle: HICON;
    function GetName: string; override;
    {* ���� GetName ���������� WizardName ��Ϊ��ʾ���ַ�����  }
  public
    constructor Create; override;
    {* �๹���� }
    destructor Destroy; override;
    {* �������� }

    // IOTARepositoryWizard methods
    function GetPage: string;
    {$IFDEF COMPILER6_UP}
    function GetGlyph: Cardinal;
    {$ELSE}
    function GetGlyph: HICON;
    {$ENDIF}
  end;

//==============================================================================
// ��Ԫģ���򵼻���
//==============================================================================

{ TCnUnitWizard }

  TCnUnitWizard = class(TCnRepositoryWizard, {$IFDEF DELPHI10_UP}IOTAProjectWizard{$ELSE}IOTAFormWizard{$ENDIF});
  {* ����ʵ�� IOTAFormWizard ������ New �Ի����г���, BDS2006 ��Ҫ�� IOTAProjectWizard}

//==============================================================================
// ����ģ���򵼻���
//==============================================================================

{ TCnFormWizard }

  TCnFormWizard = class(TCnRepositoryWizard, IOTAFormWizard);

//==============================================================================
// ����ģ���򵼻���
//==============================================================================

{ TCnProjectWizard }

  TCnProjectWizard = class(TCnRepositoryWizard, IOTAProjectWizard);

//==============================================================================
// �������༭���Ҽ��˵�ִ����Ŀ�Ļ��࣬�����������Ӧ����ʵ�ֹ���
//==============================================================================

{ TCnBaseMenuExecutor }

  TCnBaseMenuExecutor = class(TObject)
  {* �������༭���Ҽ��˵�ִ����Ŀ�Ļ��࣬�ɴ�����ĳһר��ʵ��}
  private
    FTag: Integer;
    FWizard: TCnBaseWizard;
  public
    constructor Create(OwnWizard: TCnBaseWizard); virtual;
    {* �๹���� }
    destructor Destroy; override;
    {* �������� }

    function GetActive: Boolean; virtual;
    {* ������Ŀ�Ƿ���ʾ������˳���ŵ���}
    function GetCaption: string; virtual;
    {* ��Ŀ��ʾ�ı��⣬����˳���ŵ�һ}
    function GetHint: string; virtual;
    {* ��Ŀ����ʾ}
    function GetEnabled: Boolean; virtual;
    {* ������Ŀ�Ƿ�ʹ�ܣ�����˳���ŵ���}
    procedure Prepare; virtual;
    {* PrepareItem ʱ�����ã�����˳���ŵڶ�}
    function Execute: Boolean; virtual;
    {* ��Ŀִ�з���������Ĭ��ʲô������}

    property Wizard: TCnBaseWizard read FWizard;
    {* ���� Wizard ʵ��}
    property Tag: Integer read FTag write FTag;
    {* ��һ�� Tag}
  end;

//==============================================================================
// �������༭���Ҽ��˵�ִ����Ŀ����һ��ʽ�Ļ��࣬�����������¼���ָ��ִ�в���
//==============================================================================

{ TCnContextMenuExecutor }

  TCnContextMenuExecutor = class(TCnBaseMenuExecutor)
  {* �������༭���Ҽ��˵�ִ����Ŀ����һ��ʽ�Ļ��࣬�����������¼���ָ��ִ�в���}
  private
    FActive: Boolean;
    FEnabled: Boolean;
    FCaption: string;
    FHint: string;
    FOnExecute: TNotifyEvent;
  protected
    procedure DoExecute; virtual;
  public
    constructor Create; reintroduce; virtual;

    function GetActive: Boolean; override;
    function GetCaption: string; override;
    function GetHint: string; override;
    function GetEnabled: Boolean; override;
    function Execute: Boolean; override;

    property Caption: string read FCaption write FCaption;
    {* ��Ŀ��ʾ�ı���}
    property Hint: string read FHint write FHint;
    {* ��Ŀ��ʾ����ʾ}
    property Active: Boolean read FActive write FActive;
    {* ������Ŀ�Ƿ���ʾ}
    property Enabled: Boolean read FEnabled write FEnabled;
    {* ������Ŀ�Ƿ�ʹ��}
    property OnExecute: TNotifyEvent read FOnExecute write FOnExecute;
    {* ��Ŀִ�з�����ִ��ʱ����}
  end;

//==============================================================================
// ר�����б���ع���
//==============================================================================

procedure RegisterCnWizard(const AClass: TCnWizardClass);
{* ע��һ�� CnBaseWizard ר�������ã�ÿ��ר��ʵ�ֵ�ԪӦ�ڸõ�Ԫ�� initialization
   �ڵ��øù���ע��ר���� }

function GetCnWizardClass(const ClassName: string): TCnWizardClass;
{* ����ר������ȡָ����ר�������� }

function GetCnWizardClassCount: Integer;
{* ������ע���ר�������� }

function GetCnWizardClassByIndex(const Index: Integer): TCnWizardClass;
{* ����������ȡָ����ר�������� }

function GetCnWizardTypeNameFromClass(AClass: TClass): string;
{* ����ר������ȡר���������� }

function GetCnWizardTypeName(AWizard: TCnBaseWizard): string;
{* ����ר��ʵ����ȡָ����ר�������� }

procedure GetCnWizardInfoStrs(AWizard: TCnBaseWizard; Infos: TStrings);
{* ��ȡר��ʵ���������ַ����б�����Ϣ�����}

implementation

uses
  CnWizUtils, CnWizOptions, CnCommon, CnWizCommentFrm, CnWizSubActionShortCutFrm;

procedure RegisterCnWizard(const AClass: TCnWizardClass);
begin
end;

function GetCnWizardClass(const ClassName: string): TCnWizardClass;
begin
end;

function GetCnWizardClassCount: Integer;
begin
end;

function GetCnWizardClassByIndex(const Index: Integer): TCnWizardClass;
begin
end;

function GetCnWizardTypeNameFromClass(AClass: TClass): string;
begin
end;

function GetCnWizardTypeName(AWizard: TCnBaseWizard): string;
begin
end;

procedure GetCnWizardInfoStrs(AWizard: TCnBaseWizard; Infos: TStrings);
begin
end;

constructor TCnBaseWizard.Create;
begin
end;

destructor TCnBaseWizard.Destroy;
begin
end;

class procedure TCnBaseWizard.GetWizardInfo(var Name, Author, Email,
  Comment: string);
begin
end;

class function TCnBaseWizard.WizardName: string;
begin
end;

class function TCnBaseWizard.GetIDStr: string;
begin
end;

function TCnBaseWizard.GetAuthor: string;
begin
end;

function TCnBaseWizard.GetComment: string;
begin
end;

function TCnBaseWizard.GetSearchContent: string;
begin
end;

class function TCnBaseWizard.IsInternalWizard: Boolean;
begin
end;

procedure TCnBaseWizard.DebugComand(Cmds: TStrings; Results: TStrings);
begin
end;

function TCnBaseWizard.CreateIniFile(CompilerSection: Boolean): TCustomIniFile;
begin
end;

procedure TCnBaseWizard.DoLoadSettings;
begin
end;

procedure TCnBaseWizard.DoSaveSettings;
begin
end;

procedure TCnBaseWizard.DoResetSettings;
begin
end;

procedure TCnBaseWizard.Config;
begin
end;

procedure TCnBaseWizard.LanguageChanged(Sender: TObject);
begin
end;

procedure TCnBaseWizard.LoadSettings(Ini: TCustomIniFile);
begin
end;

procedure TCnBaseWizard.SaveSettings(Ini: TCustomIniFile);
begin
end;

procedure TCnBaseWizard.ResetSettings(Ini: TCustomIniFile);
begin
end;

procedure TCnBaseWizard.Loaded;
begin
end;

procedure TCnBaseWizard.LaterLoaded;
begin
end;

function TCnBaseWizard.GetHasConfig: Boolean;
begin
end;

procedure TCnBaseWizard.SetActive(Value: Boolean);
begin
end;

function TCnBaseWizard.GetIDString: string;
begin
end;

function TCnBaseWizard.GetName: string;
begin
end;

function TCnBaseWizard.GetState: TWizardState;
begin
end;

constructor TCnIconWizard.Create;
begin
end;

destructor TCnIconWizard.Destroy;
begin
end;

function TCnIconWizard.GetIcon: TIcon;
begin
end;

function TCnIconWizard.GetBigIcon: TIcon;
begin
end;

class function TCnIconWizard.GetIconName: string;
begin
end;

procedure TCnIconWizard.InitIcon(AIcon, ASmallIcon: TIcon);
begin
end;

constructor TCnActionWizard.Create;
begin
end;

destructor TCnActionWizard.Destroy;
begin
end;

procedure TCnActionWizard.RefreshAction;
begin
end;

procedure TCnActionWizard.Click(Sender: TObject);
begin
end;

procedure TCnActionWizard.OnActionUpdate(Sender: TObject);
begin
end;

function TCnActionWizard.CreateAction: TCnWizAction;
begin
end;

function TCnActionWizard.GetDefShortCut: TShortCut;
begin
end;

function TCnActionWizard.EnableShortCut: Boolean;
begin
end;

function TCnActionWizard.GetHint: string;
begin
end;

function TCnActionWizard.GetSearchContent: string;
begin
end;

function TCnActionWizard.GetIcon: TIcon;
begin
end;

function TCnActionWizard.GetImageIndex: Integer;
begin
end;

constructor TCnMenuWizard.Create;
begin
end;

procedure TCnMenuWizard.SetMenuOrder(const Value: Integer);
begin
end;

function TCnMenuWizard.CreateAction: TCnWizAction;
begin
end;

function TCnMenuWizard.GetAction: TCnWizMenuAction;
begin
end;

function TCnMenuWizard.GetMenu: TMenuItem;
begin
end;

function TCnMenuWizard.EnableShortCut: Boolean;
begin
end;

constructor TCnSubMenuWizard.Create;
begin
end;

destructor TCnSubMenuWizard.Destroy;
begin
end;

function TCnSubMenuWizard.GetSearchContent: string;
begin
end;

procedure TCnSubMenuWizard.DebugComand(Cmds: TStrings; Results: TStrings);
begin
end;

procedure TCnSubMenuWizard.AcquireSubActions;
begin
end;

function TCnSubMenuWizard.CreateAction: TCnWizAction;
begin
end;

procedure TCnSubMenuWizard.Execute;
begin
end;

function TCnSubMenuWizard.EnableShortCut: Boolean;
begin
end;

procedure TCnSubMenuWizard.RefreshAction;
begin
end;

procedure TCnSubMenuWizard.RefreshSubActions;
begin
end;

function TCnSubMenuWizard.RegisterASubAction(const ACommand, ACaption: string;
  AShortCut: TShortCut; const AHint: string; const AIconName: string): Integer;
begin
end;

procedure TCnSubMenuWizard.AddSubMenuWizard(SubMenuWiz: TCnSubMenuWizard);
begin
end;

procedure TCnSubMenuWizard.AddSepMenu;
begin
end;

procedure TCnSubMenuWizard.ClearSubActions;
begin
end;

procedure TCnSubMenuWizard.DeleteSubAction(Index: Integer);
begin
end;

procedure TCnSubMenuWizard.FreeSubMenus;
begin
end;

function TCnSubMenuWizard.IndexOf(SubAction: TCnWizMenuAction): Integer;
begin
end;

function TCnSubMenuWizard.ActionByCommand(const ACommand: string): TCnWizAction;
begin
end;

procedure TCnSubMenuWizard.Click(Sender: TObject);
begin
end;

procedure TCnSubMenuWizard.OnExecute(Sender: TObject);
begin
end;

procedure TCnSubMenuWizard.OnUpdate(Sender: TObject);
begin
end;

function TCnSubMenuWizard.ShowShortCutDialog(const HelpStr: string): Boolean;
begin
end;

procedure TCnSubMenuWizard.SetActive(Value: Boolean);
begin
end;

procedure TCnSubMenuWizard.OnActionUpdate(Sender: TObject);
begin
end;

procedure TCnSubMenuWizard.OnPopup(Sender: TObject);
begin
end;

procedure TCnSubMenuWizard.SubActionExecute(Index: Integer);
begin
end;

procedure TCnSubMenuWizard.SubActionUpdate(Index: Integer);
begin
end;

function TCnSubMenuWizard.GetSubActionCount: Integer;
begin
end;

function TCnSubMenuWizard.GetSubActions(Index: Integer): TCnWizMenuAction;
begin
end;

function TCnSubMenuWizard.GetSubMenus(Index: Integer): TMenuItem;
begin
end;

constructor TCnRepositoryWizard.Create;
begin
end;

destructor TCnRepositoryWizard.Destroy;
begin
end;

function TCnRepositoryWizard.GetName: string;
begin
end;

function TCnRepositoryWizard.GetGlyph: Cardinal;
begin
end;

function TCnRepositoryWizard.GetPage: string;
begin
end;

constructor TCnBaseMenuExecutor.Create(OwnWizard: TCnBaseWizard);
begin
end;

destructor TCnBaseMenuExecutor.Destroy;
begin
end;

procedure TCnBaseMenuExecutor.Prepare;
begin
end;

function TCnBaseMenuExecutor.Execute: Boolean;
begin
end;

function TCnBaseMenuExecutor.GetActive: Boolean;
begin
end;

function TCnBaseMenuExecutor.GetCaption: string;
begin
end;

function TCnBaseMenuExecutor.GetEnabled: Boolean;
begin
end;

function TCnBaseMenuExecutor.GetHint: string;
begin
end;

constructor TCnContextMenuExecutor.Create;
begin
end;

procedure TCnContextMenuExecutor.DoExecute;
begin
end;

function TCnContextMenuExecutor.Execute: Boolean;
begin
end;

function TCnContextMenuExecutor.GetActive: Boolean;
begin
end;

function TCnContextMenuExecutor.GetCaption: string;
begin
end;

function TCnContextMenuExecutor.GetEnabled: Boolean;
begin
end;

function TCnContextMenuExecutor.GetHint: string;
begin
end;

end.
