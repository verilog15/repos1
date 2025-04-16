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

{******************************************************************************}
{ Unit Note:                                                                   }
{    This file is partly derived from GExperts 1.2                             }
{                                                                              }
{ Original author:                                                             }
{    GExperts, Inc  http://www.gexperts.org/                                   }
{    Erik Berry <eberry@gexperts.org> or <eb@techie.com>                       }
{******************************************************************************}

unit CnWizIdeUtils;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��ڽű���ʹ�õ� CnIdeWizUtils ��Ԫ����
* ��Ԫ���ߣ�CnPack ������
* ��    ע������Ԫ�����������ͺͺ��������� PasScript �ű���ʹ��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2006.12.31 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

uses
  Windows, Messages, Classes, Controls, SysUtils, Graphics, Forms, ComCtrls,
  ExtCtrls, Menus, Buttons, Tabs,
{$IFNDEF VER130}
  DesignIntf,
{$ENDIF}
  ToolsAPI;

type
  TCnModuleSearchType = (mstInvalid, mstProject, mstProjectSearch, mstSystemSearch);
  {* ��������Դ��λ�����ͣ��Ƿ��������ڡ���������Ŀ¼�ڡ�ϵͳ����Ŀ¼��}

//==============================================================================
// IDE ����༭�����ܺ���
//==============================================================================

function IdeGetEditorSelectedLines(Lines: TStringList): Boolean;
{* ȡ�õ�ǰ����༭��ѡ���еĴ��룬ʹ������ģʽ�����ѡ���Ϊ�գ��򷵻ص�ǰ�д��롣}

function IdeGetEditorSelectedText(Lines: TStringList): Boolean;
{* ȡ�õ�ǰ����༭��ѡ���Ĵ��롣}

function IdeGetEditorSourceLines(Lines: TStringList): Boolean;
{* ȡ�õ�ǰ����༭��ȫ��Դ���롣}

function IdeSetEditorSelectedLines(Lines: TStringList): Boolean;
{* �滻��ǰ����༭��ѡ���еĴ��룬ʹ������ģʽ�����ѡ���Ϊ�գ����滻��ǰ�д��롣}

function IdeSetEditorSelectedText(Lines: TStringList): Boolean;
{* �滻��ǰ����༭��ѡ���Ĵ��롣}

function IdeSetEditorSourceLines(Lines: TStringList): Boolean;
{* �滻��ǰ����༭��ȫ��Դ���롣}

function IdeInsertTextIntoEditor(const Text: string): Boolean;
{* �����ı�����ǰ�༭����֧�ֶ����ı���}

function IdeEditorGetEditPos(var Col, Line: Integer): Boolean;
{* ���ص�ǰ���λ�ã���� EditView Ϊ��ʹ�õ�ǰֵ�� }

function IdeEditorGotoEditPos(Col, Line: Integer; Middle: Boolean): Boolean;
{* �ƶ���굽ָ��λ�ã�Middle ��ʾ�Ƿ��ƶ���ͼ�����ġ�}

function IdeGetBlockIndent: Integer;
{* ��õ�ǰ�༭����������� }

function IdeGetSourceByFileName(const FileName: string): string;
{* �����ļ���ȡ�����ݡ�����ļ��� IDE �д򿪣����ر༭���е����ݣ����򷵻��ļ����ݡ�}

function IdeSetSourceByFileName(const FileName: string; Source: TStrings;
  OpenInIde: Boolean): Boolean;
{* �����ļ���д�����ݡ�����ļ��� IDE �д򿪣�д�����ݵ��༭���У��������
   OpenInIde Ϊ����ļ�д�뵽�༭����OpenInIde Ϊ��ֱ��д���ļ���}

//==============================================================================
// IDE ����༭�����ܺ���
//==============================================================================

function IdeGetFormDesigner(FormEditor: IOTAFormEditor = nil): IDesigner;
{* ȡ�ô���༭�����������FormEditor Ϊ nil ��ʾȡ��ǰ���� }

function IdeGetDesignedForm(Designer: IDesigner = nil): TCustomForm;
{* ȡ�õ�ǰ��ƵĴ��� }

function IdeGetFormSelection(Selections: TList; Designer: IDesigner = nil;
  ExcludeForm: Boolean = True): Boolean;
{* ȡ�õ�ǰ��ƴ�������ѡ������ }
 
//==============================================================================
// �޸��� GExperts Src 1.12 �� IDE ��غ���
//==============================================================================

function GetIdeMainForm: TCustomForm;
{* ���� IDE ������ (TAppBuilder) }

function GetIdeEdition: string;
{* ���� IDE �汾}

function GetComponentPaletteTabControl: TTabControl;
{* ������������󣬿���Ϊ�գ�ֻ֧�� 2010 ���°汾}

function GetNewComponentPaletteTabControl: TWinControl;
{* ���� 2010 �����ϵ����������ϰ벿�� Tab ���󣬿���Ϊ��}

function GetNewComponentPaletteComponentPanel: TWinControl;
{* ���� 2010 �����ϵ����������°벿����������б���������󣬿���Ϊ��}

function GetObjectInspectorForm: TCustomForm;
{* ���ض����������壬����Ϊ��}

function GetComponentPalettePopupMenu: TPopupMenu;
{* �����������Ҽ��˵�������Ϊ��}

function GetComponentPaletteControlBar: TControlBar;
{* �������������ڵ�ControlBar������Ϊ��}

function GetMainMenuItemHeight: Integer;
{* �������˵���߶� }

function IsIdeEditorForm(AForm: TCustomForm): Boolean;
{* �ж�ָ�������Ƿ�༭������}

function IsIdeDesignForm(AForm: TCustomForm): Boolean;
{* �ж�ָ�������Ƿ�������ڴ���}

procedure BringIdeEditorFormToFront;
{* ��Դ��༭����Ϊ��Ծ}

function IDEIsCurrentWindow: Boolean;
{* �ж� IDE �Ƿ��ǵ�ǰ�Ļ���� }

//==============================================================================
// ������ IDE ��غ���
//==============================================================================

function GetInstallDir: string;
{* ȡ��������װĿ¼}

{$IFDEF BDS}
function GetBDSUserDataDir: string;
{* ȡ�� BDS (Delphi8/9) ���û�����Ŀ¼ }
{$ENDIF}

procedure GetProjectLibPath(Paths: TStrings);
{* ȡ��ǰ���������� Path ����}

function GetFileNameFromModuleName(AName: string; AProject: IOTAProject = nil): string;
{* ����ģ������������ļ���}

function CnOtaGetVersionInfoKeys(Project: IOTAProject = nil): TStrings;
{* ��ȡ��ǰ��Ŀ�еİ汾��Ϣ��ֵ}

procedure GetLibraryPath(Paths: TStrings; IncludeProjectPath: Boolean = True);
{* ȡ���������е� LibraryPath ����}

function GetComponentUnitName(const ComponentName: string): string;
{* ȡ����������ڵĵ�Ԫ��}

procedure GetInstalledComponents(Packages, Components: TStrings);
{* ȡ�Ѱ�װ�İ����������������Ϊ nil�����ԣ�}

function GetEditControlFromEditorForm(AForm: TCustomForm): TControl;
{* ���ر༭�����ڵı༭���ؼ� }

function GetCurrentEditControl: TControl;
{* ���ص�ǰ�Ĵ���༭���ؼ� }

function GetStatusBarFromEditor(EditControl: TControl): TStatusBar;
{* �ӱ༭���ؼ�����������ı༭�����ڵ�״̬��}

function GetCurrentSyncButton: TControl;
{* ��ȡ��ǰ��ǰ�˱༭�����﷨�༭��ť��ע���﷨�༭��ť���ڲ����ڿɼ�}

function GetCurrentSyncButtonVisible: Boolean;
{* ��ȡ��ǰ��ǰ�˱༭�����﷨�༭��ť�Ƿ�ɼ����ް�ť�򲻿ɼ������� False}

function GetCodeTemplateListBox: TControl;
{* ���ر༭���еĴ���ģ���Զ������}

function GetCodeTemplateListBoxVisible: Boolean;
{* ���ر༭���еĴ���ģ���Զ�������Ƿ�ɼ����޻򲻿ɼ������� False}

function IsCurrentEditorInSyncMode: Boolean;
{* ��ǰ�༭���Ƿ����﷨��༭ģʽ�£���֧�ֻ��ڿ�ģʽ�·��� False}

function IsKeyMacroRunning: Boolean;
{* ��ǰ�Ƿ��ڼ��̺��¼�ƻ�طţ���֧�ֻ��ڷ��� False}

function GetCurrentCompilingProject: IOTAProject;
{* ���ص�ǰ���ڱ���Ĺ��̣�ע�ⲻһ���ǵ�ǰ����}

function CompileProject(AProject: IOTAProject): Boolean;
{* ���빤�̣����ر����Ƿ�ɹ�}

//==============================================================================
// �������װ��
//==============================================================================

type

{ TCnPaletteWrapper }

  TCnPaletteWrapper = class(TObject)
  private
    function GetActiveTab: string;
    function GetEnabled: Boolean;
    function GetIsMultiLine: Boolean;
    function GetPalToolCount: Integer;
    function GetSelectedIndex: Integer;
    function GetSelectedToolName: string;
    function GetSelector: TSpeedButton;
    function GetTabCount: Integer;
    function GetTabIndex: Integer;
    function GetTabs(Index: Integer): string;
    function GetVisible: Boolean;
    procedure SetEnabled(const Value: Boolean);
    procedure SetSelectedIndex(const Value: Integer);
    procedure SetTabIndex(const Value: Integer);
    procedure SetVisible(const Value: Boolean);

  public
    constructor Create;

    procedure BeginUpdate;
    {* ��ʼ���£���ֹˢ��ҳ�� }
    procedure EndUpdate;
    {* ֹͣ���£��ָ�ˢ��ҳ�� }
    function SelectComponent(const AComponent: string; const ATab: string): Boolean;
    {* ��������ѡ�пؼ����е�ĳ�ؼ� }
    function FindTab(const ATab: string): Integer;
    {* ����ĳҳ������� }
    property SelectedIndex: Integer read GetSelectedIndex write SetSelectedIndex;
    {* ���µĿؼ��ڱ�ҳ����ţ�0 ��ͷ }
    property SelectedToolName: string read GetSelectedToolName;
    {* ���µĿؼ���������δ������Ϊ�� }
    property Selector: TSpeedButton read GetSelector;
    {* �����л��������� SpeedButton }
    property PalToolCount: Integer read GetPalToolCount;
    {* ��ǰҳ�ؼ����� }
    property ActiveTab: string read GetActiveTab;
    {* ��ǰҳ���� }
    property TabIndex: Integer read GetTabIndex write SetTabIndex;
    {* ��ǰҳ���� }
    property Tabs[Index: Integer]: string read GetTabs;
    {* ���������õ�ҳ���� }
    property TabCount: Integer read GetTabCount;
    {* �ؼ�����ҳ�� }
    property IsMultiLine: Boolean read GetIsMultiLine;
    {* �ؼ����Ƿ���� }
    property Visible: Boolean read GetVisible write SetVisible;
    {* �ؼ����Ƿ�ɼ� }
    property Enabled: Boolean read GetEnabled write SetEnabled;
    {* �ؼ����Ƿ�ʹ�� }
  end;

{ TCnMessageViewWrapper }

{$IFDEF BDS}
  TXTreeView = TCustomControl;
{$ELSE}
  TXTreeView = TTreeView;
{$ENDIF BDS}

  TCnMessageViewWrapper = class(TObject)
  {* ��װ����Ϣ��ʾ���ڵĸ������Ե��� }
  private
    FMessageViewForm: TCustomForm;
    FEditMenuItem: TMenuItem;
    FTabSet: TTabSet;
    FTreeView: TXTreeView;
{$IFNDEF BDS}
    function GetMessageCount: Integer;
    function GetSelectedIndex: Integer;
    procedure SetSelectedIndex(const Value: Integer);
    function GetCurrentMessage: string;
{$ENDIF}
    function GetTabCaption: string;
    function GetTabCount: Integer;
    function GetTabIndex: Integer;
    procedure SetTabIndex(const Value: Integer);
    function GetTabSetVisible: Boolean;
  public
    constructor Create;

    procedure UpdateAllItems;

    procedure EditMessageSource;
    {* ˫����Ϣ����}

    property MessageViewForm: TCustomForm read FMessageViewForm;
    {* ��Ϣ����}
    property TreeView: TXTreeView read FTreeView;
    {* ��Ϣ�����ʵ����BDS �·� TreeView�����ֻ�ܷ��� CustomControl }
{$IFNDEF BDS}
    property SelectedIndex: Integer read GetSelectedIndex write SetSelectedIndex;
    {* ��Ϣ��ѡ�е����}
    property MessageCount: Integer read GetMessageCount;
    {* ���е���Ϣ��}
    property CurrentMessage: string read GetCurrentMessage;
    {* ��ǰѡ�е���Ϣ�����ƺ����Ƿ��ؿ�}
{$ENDIF}
    property TabSet: TTabSet read FTabSet;
    {* ���ط�ҳ�����ʵ��}
    property TabSetVisible: Boolean read GetTabSetVisible;
    {* ���ط�ҳ����Ƿ�ɼ���D5 ��Ĭ�ϲ��ɼ�}
    property TabIndex: Integer read GetTabIndex write SetTabIndex;
    {* ����/���õ�ǰҳ���}
    property TabCount: Integer read GetTabCount;
    {* ������ҳ��}
    property TabCaption: string read GetTabCaption;
    {* ���ص�ǰҳ���ַ���}
    property EditMenuItem: TMenuItem read FEditMenuItem;
    {* '�༭'�˵���}
  end;

{ TCnEditControlWrapper }

  TCnEditControlInfo = record
  {* ����༭��λ����Ϣ }
    TopLine: Integer;         // ���к�
    LinesInWindow: Integer;   // ������ʾ����
    LineCount: Integer;       // ���뻺����������
    CaretX: Integer;          // ���Xλ��
    CaretY: Integer;          // ���Yλ��
    CharXIndex: Integer;      // �ַ����
{$IFDEF BDS}
    LineDigit: Integer;       // �༭����������λ������100��Ϊ3, �������
{$ENDIF}
  end;

  TEditorChangeType = (
    ctView,                   // ��ǰ��ͼ�л�
    ctWindow,                 // �������С�β�б仯
    ctCurrLine,               // ��ǰ�����
    ctCurrCol,                // ��ǰ�����
    ctFont,                   // ������
    ctVScroll,                // �༭����ֱ����
    ctHScroll,                // �༭���������
    ctBlock,                  // ����
    ctModified,               // �༭�����޸�
    ctTopEditorChanged,       // ��ǰ��ʾ���ϲ�༭�����
{$IFDEF BDS}
    ctLineDigit,              // �༭��������λ���仯����99��100
{$ENDIF}
    ctElided,                 // �༭�����۵�������֧��
    ctUnElided,               // �༭����չ��������֧��
    ctOptionChanged           // �༭�����öԻ��������򿪹�
    );

  TEditorChangeTypes = set of TEditorChangeType;

  TCnEditorContext = record
    TopRow: Integer;               // �Ӿ��ϵ�һ�е��к�
    BottomRow: Integer;            // �Ӿ���������һ�е��к�
    LeftColumn: Integer;
    CurPos: TOTAEditPos;
    LineCount: Integer;            // ��¼�༭���������������
    LineText: string;
    ModTime: TDateTime;
    BlockValid: Boolean;
    BlockSize: Integer;
    BlockStartingColumn: Integer;
    BlockStartingRow: Integer;
    BlockEndingColumn: Integer;
    BlockEndingRow: Integer;
    EditView: Pointer;
{$IFDEF BDS}
    LineDigit: Integer;       // �༭����������λ������100��Ϊ3, �������
{$ENDIF}
  end;

  TEditorObject = class
  private
    FLines: TList;
    FLastTop: Integer;
    FLastBottomElided: Boolean;
    FLinesChanged: Boolean;
    FTopControl: TControl;
    FContext: TCnEditorContext;
    FEditControl: TControl;
    FEditWindow: TCustomForm;
    FEditView: IOTAEditView;
    FGutterWidth: Integer;
    FGutterChanged: Boolean;
    FLastValid: Boolean;
    procedure SetEditView(AEditView: IOTAEditView);
    function GetGutterWidth: Integer;
    function GetViewLineNumber(Index: Integer): Integer;
    function GetViewLineCount: Integer;
    function GetViewBottomLine: Integer;
    function GetTopEditor: TControl;
  public
    constructor Create(AEditControl: TControl; AEditView: IOTAEditView);
    destructor Destroy; override;
    function EditorIsOnTop: Boolean;
    procedure IDEShowLineNumberChanged;
    property Context: TCnEditorContext read FContext;
    property EditControl: TControl read FEditControl;
    property EditWindow: TCustomForm read FEditWindow;
    property EditView: IOTAEditView read FEditView;
    property GutterWidth: Integer read GetGutterWidth;

    // ��ǰ��ʾ����ǰ��ı༭�ؼ�
    property TopControl: TControl read FTopControl;
    // ��ͼ����Ч����
    property ViewLineCount: Integer read GetViewLineCount;
    // ��ͼ����ʾ����ʵ�кţ�Index �� 0 ��ʼ
    property ViewLineNumber[Index: Integer]: Integer read GetViewLineNumber;
    // ��ͼ����ʾ�������ʵ�к�
    property ViewBottomLine: Integer read GetViewBottomLine;
  end;

  THighlightItem = class
  {* ��ͬ�༭��Ԫ�صĸ�����ʾ���ԣ���������������}
  private
    FBold: Boolean;
    FColorBk: TColor;
    FColorFg: TColor;
    FItalic: Boolean;
    FUnderline: Boolean;
  public
    property Bold: Boolean read FBold write FBold;
    property ColorBk: TColor read FColorBk write FColorBk;
    property ColorFg: TColor read FColorFg write FColorFg;
    property Italic: Boolean read FItalic write FItalic;
    property Underline: Boolean read FUnderline write FUnderline;
  end;

  TEditorPaintLineNotifier = procedure (Editor: TEditorObject;
    LineNum, LogicLineNum: Integer) of object;
  {* EditControl �ؼ����л���֪ͨ�¼����û����Դ˽����Զ������}

  TEditorPaintNotifier = procedure (EditControl: TControl; EditView: IOTAEditView)
    of object;
  {* EditControl �ؼ���������֪ͨ�¼����û����Դ˽����Զ������}

  TEditorNotifier = procedure (EditControl: TControl; EditWindow: TCustomForm;
    Operation: TOperation) of object;
  {* �༭��������ɾ��֪ͨ}

  TEditorChangeNotifier = procedure (Editor: TEditorObject; ChangeType:
    TEditorChangeTypes) of object;
  {* �༭�����֪ͨ}

  TKeyMessageNotifier = procedure (Key, ScanCode: Word; Shift: TShiftState;
    var Handled: Boolean) of object;
  {* �����¼�}

  // ����¼������� TControl �ڵĶ��壬�� Sender �� TEditorObject�����Ҽ����Ƿ��Ƿǿͻ����ı�־
  TEditorMouseUpNotifier = procedure(Editor: TEditorObject; Button: TMouseButton;
    Shift: TShiftState; X, Y: Integer; IsNC: Boolean) of object;
  {* �༭�������̧��֪ͨ}

  TEditorMouseDownNotifier =  procedure(Editor: TEditorObject; Button: TMouseButton;
    Shift: TShiftState; X, Y: Integer; IsNC: Boolean) of object;
  {* �༭������갴��֪ͨ}

  TEditorMouseMoveNotifier = procedure(Editor: TEditorObject; Shift: TShiftState;
    X, Y: Integer; IsNC: Boolean) of object;
  {* �༭��������ƶ�֪ͨ}

  TEditorMouseLeaveNotifier = procedure(Editor: TEditorObject; IsNC: Boolean) of object;
  {* �༭��������뿪֪ͨ}

  // �༭���ǿͻ������֪ͨ�����ڹ������ػ�
  TEditorNcPaintNotifier = procedure(Editor: TEditorObject) of object;
  {* �༭���ǿͻ����ػ�֪ͨ}

  TEditorVScrollNotifier = procedure(Editor: TEditorObject) of object;
  {* �༭���������֪ͨ}

  TCnBreakPointClickItem = class
  private
    FBpPosY: Integer;
    FBpDeltaLine: Integer;
    FBpEditView: IOTAEditView;
    FBpEditControl: TControl;
  public
    property BpEditControl: TControl read FBpEditControl write FBpEditControl;
    property BpEditView: IOTAEditView read FBpEditView write FBpEditView;
    property BpPosY: Integer read FBpPosY write FBpPosY;
    property BpDeltaLine: Integer read FBpDeltaLine write FBpDeltaLine;
  end;

  TCnEditControlWrapper = class(TComponent)
  private
    FCorIdeModule: HMODULE;
    FAfterPaintLineNotifiers: TList;
    FBeforePaintLineNotifiers: TList;
    FEditControlNotifiers: TList;
    FEditorChangeNotifiers: TList;
    FKeyDownNotifiers: TList;
    FKeyUpNotifiers: TList;
    FCharSize: TSize;
    FHighlights: TStringList;
    FPaintNotifyAvailable: Boolean;
    FMouseNotifyAvailable: Boolean;
    FPaintLineHook: TCnMethodHook;
    FSetEditViewHook: TCnMethodHook;

    FMouseUpNotifiers: TList;
    FMouseDownNotifiers: TList;
    FMouseMoveNotifiers: TList;
    FMouseLeaveNotifiers: TList;
    FNcPaintNotifiers: TList;
    FVScrollNotifiers: TList;

    FEditorList: TObjectList;
    FEditControlList: TList;
    FOptionChanged: Boolean;
    FOptionDlgVisible: Boolean;
    FSaveFontName: string;
    FSaveFontSize: Integer;
{$IFDEF IDE_HAS_ERRORINSIGHT}
    FSaveErrorInsightIsSmoothWave: Boolean;
{$ENDIF}
    FFontArray: array[0..9] of TFont;

    FBpClickQueue: TQueue;
    FEditorBaseFont: TFont;
    procedure ScrollAndClickEditControl(Sender: TObject);

    procedure AddNotifier(List: TList; Notifier: TMethod);
    function CalcCharSize: Boolean;
    // �����ַ����ߴ磬����˼���Ǵ�ע������ø��ָ������ü��㣬ȡ�����
    procedure GetHighlightFromReg;
    procedure ClearAndFreeList(var List: TList);
    function IndexOf(List: TList; Notifier: TMethod): Integer;
    procedure InitEditControlHook;
    procedure CheckAndSetEditControlMouseHookFlag;
    procedure RemoveNotifier(List: TList; Notifier: TMethod);
    function UpdateCharSize: Boolean;
    procedure EditControlProc(EditWindow: TCustomForm; EditControl:
      TControl; Context: Pointer);
    procedure UpdateEditControlList;
    procedure CheckOptionDlg;
    function GetEditorContext(Editor: TEditorObject): TCnEditorContext;
    function CheckViewLines(Editor: TEditorObject; Context: TCnEditorContext): Boolean;
    function CheckEditorChanges(Editor: TEditorObject): TEditorChangeTypes;
    procedure OnActiveFormChange(Sender: TObject);
    procedure AfterThemeChange(Sender: TObject);
    procedure OnSourceEditorNotify(SourceEditor: IOTASourceEditor;
      NotifyType: TCnWizSourceEditorNotifyType; EditView: IOTAEditView);
    procedure ApplicationMessage(var Msg: TMsg; var Handled: Boolean);
    procedure OnCallWndProcRet(Handle: HWND; Control: TWinControl; Msg: TMessage);
    procedure OnGetMsgProc(Handle: HWND; Control: TWinControl; Msg: TMessage);
    procedure OnIdle(Sender: TObject);
    function GetEditorCount: Integer;
    function GetEditors(Index: Integer): TEditorObject;
    function GetHighlight(Index: Integer): THighlightItem;
    function GetHighlightCount: Integer;
    function GetHighlightName(Index: Integer): string;
    procedure ClearHighlights;
    procedure LoadFontFromRegistry;
    procedure ResetFontsFromBasic(ABasicFont: TFont);
    function GetFonts(Index: Integer): TFont;
    procedure SetFonts(const Index: Integer; const Value: TFont);
  protected
    procedure DoAfterPaintLine(Editor: TEditorObject; LineNum, LogicLineNum: Integer);
    procedure DoBeforePaintLine(Editor: TEditorObject; LineNum, LogicLineNum: Integer);
    procedure DoAfterElide(EditControl: TControl);   // �ݲ�֧��
    procedure DoAfterUnElide(EditControl: TControl); // �ݲ�֧��
    procedure DoEditControlNotify(EditControl: TControl; Operation: TOperation);
    procedure DoEditorChange(Editor: TEditorObject; ChangeType: TEditorChangeTypes);

    procedure DoMouseDown(Editor: TEditorObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer; IsNC: Boolean);
    procedure DoMouseUp(Editor: TEditorObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer; IsNC: Boolean);
    procedure DoMouseMove(Editor: TEditorObject; Shift: TShiftState;
      X, Y: Integer; IsNC: Boolean);
    procedure DoMouseLeave(Editor: TEditorObject; IsNC: Boolean);
    procedure DoNcPaint(Editor: TEditorObject);
    procedure DoVScroll(Editor: TEditorObject);

    procedure Notification(AComponent: TComponent; Operation: TOperation); override;

    procedure CheckNewEditor(EditControl: TControl; View: IOTAEditView);
    function AddEditor(EditControl: TControl; View: IOTAEditView): Integer;
    procedure DeleteEditor(EditControl: TControl);
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

    function IndexOfEditor(EditControl: TControl): Integer; overload;
    function IndexOfEditor(EditView: IOTAEditView): Integer; overload;
    function GetEditorObject(EditControl: TControl): TEditorObject;
    property Editors[Index: Integer]: TEditorObject read GetEditors;
    property EditorCount: Integer read GetEditorCount;

    // ���¼����Ƿ�װ�ı༭��������ʾ�Ĳ�ͬԪ�ص����ԣ������������屾����Ҫ��� EditorBaseFont ����ʹ��
    function IndexOfHighlight(const Name: string): Integer;
    property HighlightCount: Integer read GetHighlightCount;
    property HighlightNames[Index: Integer]: string read GetHighlightName;
    property Highlights[Index: Integer]: THighlightItem read GetHighlight;

    function GetCharHeight: Integer;
    {* ���ر༭���и� }
    function GetCharWidth: Integer;
    {* ���ر༭���ֿ� }
    function GetCharSize: TSize;
    {* ���ر༭���иߺ��ֿ� }
    function GetEditControlInfo(EditControl: TControl): TCnEditControlInfo;
    {* ���ر༭����ǰ��Ϣ }
    function GetEditControlCharHeight(EditControl: TControl): Integer;
    {* ���ر༭���ڵ��ַ��߶�Ҳ�����и�}
    function GetEditControlSupportsSyntaxHighlight(EditControl: TControl): Boolean;
    {* ���ر༭���Ƿ�֧���﷨���� }
    function GetEditControlCanvas(EditControl: TControl): TCanvas;
    {* ���ر༭���Ļ�������}
    function GetEditView(EditControl: TControl): IOTAEditView;
    {* ����ָ���༭����ǰ������ EditView }
    function GetEditControl(EditView: IOTAEditView): TControl;
    {* ����ָ�� EditView ��ǰ�����ı༭�� }
    function GetTopMostEditControl: TControl;
    {* ���ص�ǰ��ǰ�˵� EditControl}
    function GetEditViewFromTabs(TabControl: TXTabControl; Index: Integer):
      IOTAEditView;
    {* ���� TabControl ָ��ҳ������ EditView }
    procedure GetAttributeAtPos(EditControl: TControl; const EdPos: TOTAEditPos;
      IncludeMargin: Boolean; var Element, LineFlag: Integer);
    {* ����ָ��λ�õĸ������ԣ������滻 IOTAEditView �ĺ��������߿��ܻᵼ�±༭�����⡣
       ��ָ��λ���ڷ� Unicode ��������� CursorPos ������D5/6/7 �� Ansi λ�ã�
       2005~2007 �� UTF8 ���ֽ�λ�ã�һ�����ֿ� 3 �У���
       2009 ����Ҫע�⣬EdPos ��ȻҲҪ���� UTF8 �ֽ�λ�á��� 2009 �� CursorPos �� Ansi��
       ����ֱ���� CursorPos ����Ϊ EdPos �����������뾭��һ�� UTF8 ת�� }
    function GetLineIsElided(EditControl: TControl; LineNum: Integer): Boolean;
    {* ����ָ�����Ƿ��۵����������۵���ͷβ��Ҳ���Ƿ����Ƿ����ء�
       ֻ�� BDS ��Ч������������� False}

    procedure ElideLine(EditControl: TControl; LineNum: Integer);
    {* �۵�ĳ�У��кű����ǿ��۵���������}
    procedure UnElideLine(EditControl: TControl; LineNum: Integer);
    {* չ��ĳ�У��кű����ǿ��۵���������}

    function GetPointFromEdPos(EditControl: TControl; APos: TOTAEditPos): TPoint;
    {* ���� BDS �б༭���ؼ�ĳ�ַ�λ�ô������ֻ꣬�� BDS ����Ч}

    function GetLineFromPoint(Point: TPoint; EditControl: TControl;
      EditView: IOTAEditView = nil): Integer;
    {* ���ر༭���ؼ�����������Ӧ���У��н����һ��ʼ������ -1 ��ʾʧ��}

    procedure MarkLinesDirty(EditControl: TControl; Line: Integer; Count: Integer);
    {* ��Ǳ༭��ָ������Ҫ�ػ棬��Ļ�ɼ���һ��Ϊ 0 }
    procedure EditorRefresh(EditControl: TControl; DirtyOnly: Boolean);
    {* ˢ�±༭�� }
    function GetTextAtLine(EditControl: TControl; LineNum: Integer): string;
    {* ȡָ���е��ı���ע��ú���ȡ�����ı��ǽ� Tab ��չ�ɿո�ģ����ʹ��
       ConvertPos ��ת���� EditPos ���ܻ������⡣ֱ�ӽ� CharIndex + 1
       ��ֵ�� EditPos.Col ���ɡ�
       �ַ����������ͣ�AnsiString/Ansi-Utf8/UnicodeString
       ���⣬LineNumΪ�߼��кţ�Ҳ���Ǻ��۵��޹ص�ʵ���кţ�1 ��ʼ }
    function IndexPosToCurPos(EditControl: TControl; Col, Line: Integer): Integer;
    {* ����༭���ַ����������༭����ʾ��ʵ��λ�� }

    procedure RepaintEditControls;
    {* ����ǿ���ñ༭���ؼ����ػ�}

    function GetUseTabKey: Boolean;
    {* ��ñ༭��ѡ���Ƿ�ʹ�� Tab ��}

    function GetTabWidth: Integer;
    {* ��ñ༭��ѡ���е� Tab �����}

    function ClickBreakpointAtActualLine(ActualLineNum: Integer; EditControl: TControl = nil): Boolean;
    {* ����༭���ؼ����ָ���еĶϵ���������/ɾ���ϵ�}

    procedure AddKeyDownNotifier(Notifier: TKeyMessageNotifier);
    {* ���ӱ༭������֪ͨ }
    procedure RemoveKeyDownNotifier(Notifier: TKeyMessageNotifier);
    {* ɾ���༭������֪ͨ }

    procedure AddKeyUpNotifier(Notifier: TKeyMessageNotifier);
    {* ���ӱ༭��������֪ͨ }
    procedure RemoveKeyUpNotifier(Notifier: TKeyMessageNotifier);
    {* ɾ���༭��������֪ͨ }

    procedure AddBeforePaintLineNotifier(Notifier: TEditorPaintLineNotifier);
    {* ���ӱ༭�������ػ�ǰ֪ͨ }
    procedure RemoveBeforePaintLineNotifier(Notifier: TEditorPaintLineNotifier);
    {* ɾ���༭�������ػ�ǰ֪ͨ }

    procedure AddAfterPaintLineNotifier(Notifier: TEditorPaintLineNotifier);
    {* ���ӱ༭�������ػ��֪ͨ }
    procedure RemoveAfterPaintLineNotifier(Notifier: TEditorPaintLineNotifier);
    {* ɾ���༭�������ػ��֪ͨ }

    procedure AddEditControlNotifier(Notifier: TEditorNotifier);
    {* ���ӱ༭��������ɾ��֪ͨ }
    procedure RemoveEditControlNotifier(Notifier: TEditorNotifier);
    {* ɾ���༭��������ɾ��֪ͨ }

    procedure AddEditorChangeNotifier(Notifier: TEditorChangeNotifier);
    {* ���ӱ༭�����֪ͨ }
    procedure RemoveEditorChangeNotifier(Notifier: TEditorChangeNotifier);
    {* ɾ���༭�����֪ͨ }

    property PaintNotifyAvailable: Boolean read FPaintNotifyAvailable;
    {* ���ر༭�����ػ�֪ͨ�����Ƿ���� }

    procedure AddEditorMouseUpNotifier(Notifier: TEditorMouseUpNotifier);
    {* ���ӱ༭�����̧��֪ͨ }
    procedure RemoveEditorMouseUpNotifier(Notifier: TEditorMouseUpNotifier);
    {* ɾ���༭�����̧��֪ͨ }

    procedure AddEditorMouseDownNotifier(Notifier: TEditorMouseDownNotifier);
    {* ���ӱ༭����갴��֪ͨ }
    procedure RemoveEditorMouseDownNotifier(Notifier: TEditorMouseDownNotifier);
    {* ɾ���༭����갴��֪ͨ }

    procedure AddEditorMouseMoveNotifier(Notifier: TEditorMouseMoveNotifier);
    {* ���ӱ༭������ƶ�֪ͨ }
    procedure RemoveEditorMouseMoveNotifier(Notifier: TEditorMouseMoveNotifier);
    {* ɾ���༭������ƶ�֪ͨ }

    procedure AddEditorMouseLeaveNotifier(Notifier: TEditorMouseLeaveNotifier);
    {* ���ӱ༭������뿪֪ͨ }
    procedure RemoveEditorMouseLeaveNotifier(Notifier: TEditorMouseLeaveNotifier);
    {* ɾ���༭������뿪֪ͨ }

    procedure AddEditorNcPaintNotifier(Notifier: TEditorNcPaintNotifier);
    {* ���ӱ༭���ǿͻ����ػ�֪ͨ }
    procedure RemoveEditorNcPaintNotifier(Notifier: TEditorNcPaintNotifier);
    {* ɾ���༭���ǿͻ����ػ�֪ͨ }

    procedure AddEditorVScrollNotifier(Notifier: TEditorVScrollNotifier);
    {* ���ӱ༭���ǿͻ����ػ�֪ͨ }
    procedure RemoveEditorVScrollNotifier(Notifier: TEditorVScrollNotifier);
    {* ɾ���༭���ǿͻ����ػ�֪ͨ }

    property MouseNotifyAvailable: Boolean read FMouseNotifyAvailable;
    {* ���ر༭��������¼�֪ͨ�����Ƿ���� }
    property EditorBaseFont: TFont read FEditorBaseFont;
    {* һ�� TFont ���󣬳��б༭���Ļ������幩���ʹ��}

    // ������ά����ע����еı༭������Ԫ�ص����壬�� Highlights ��һ���ص������ޱ���ɫ����
    property FontBasic: TFont index 0 read GetFonts write SetFonts; // ����������ǰ��ɫ
    property FontAssembler: TFont index 1 read GetFonts write SetFonts;
    property FontComment: TFont index 2 read GetFonts write SetFonts;
    property FontDirective: TFont index 3 read GetFonts write SetFonts;
    property FontIdentifier: TFont index 4 read GetFonts write SetFonts;
    property FontKeyWord: TFont index 5 read GetFonts write SetFonts;
    property FontNumber: TFont index 6 read GetFonts write SetFonts;
    property FontSpace: TFont index 7 read GetFonts write SetFonts;
    property FontString: TFont index 8 read GetFonts write SetFonts;
    property FontSymbol: TFont index 9 read GetFonts write SetFonts;
  end;

function CnPaletteWrapper: TCnPaletteWrapper;

function CnMessageViewWrapper: TCnMessageViewWrapper;

function EditControlWrapper: TCnEditControlWrapper;

implementation

{$WARNINGS OFF}

function IdeGetEditorSelectedLines(Lines: TStringList): Boolean;
begin
end;

function IdeGetEditorSelectedText(Lines: TStringList): Boolean;
begin
end;

function IdeGetEditorSourceLines(Lines: TStringList): Boolean;
begin
end;

function IdeSetEditorSelectedLines(Lines: TStringList): Boolean;
begin
end;

function IdeSetEditorSelectedText(Lines: TStringList): Boolean;
begin
end;

function IdeSetEditorSourceLines(Lines: TStringList): Boolean;
begin
end;

function IdeInsertTextIntoEditor(const Text: string): Boolean;
begin
end;

function IdeEditorGetEditPos(var Col, Line: Integer): Boolean;
begin
end;

function IdeEditorGotoEditPos(Col, Line: Integer; Middle: Boolean): Boolean;
begin
end;

function IdeGetBlockIndent: Integer;
begin
end;

function IdeGetSourceByFileName(const FileName: string): string;
begin
end;

function IdeSetSourceByFileName(const FileName: string; Source: TStrings;
  OpenInIde: Boolean): Boolean;
begin
end;

function IdeGetFormDesigner(FormEditor: IOTAFormEditor = nil): IDesigner;
begin
end;

function IdeGetDesignedForm(Designer: IDesigner = nil): TCustomForm;
begin
end;

function IdeGetFormSelection(Selections: TList; Designer: IDesigner = nil;
  ExcludeForm: Boolean = True): Boolean;
begin
end;

function GetIdeMainForm: TCustomForm;
begin
end;

function GetIdeEdition: string;
begin
end;

function GetComponentPaletteTabControl: TTabControl;
begin
end;

function GetNewComponentPaletteTabControl: TWinControl;
begin
end;

function GetNewComponentPaletteComponentPanel: TWinControl;
begin
end;

function GetObjectInspectorForm: TCustomForm;
begin
end;

function GetComponentPalettePopupMenu: TPopupMenu;
begin
end;

function GetComponentPaletteControlBar: TControlBar;
begin
end;

function GetMainMenuItemHeight: Integer;
begin
end;

function IsIdeEditorForm(AForm: TCustomForm): Boolean;
begin
end;

function IsIdeDesignForm(AForm: TCustomForm): Boolean;
begin
end;

procedure BringIdeEditorFormToFront;
begin
end;

function IDEIsCurrentWindow: Boolean;
begin
end;

function GetInstallDir: string;
begin
end;

function GetBDSUserDataDir: string;
begin
end;

procedure GetProjectLibPath(Paths: TStrings);
begin
end;

function GetFileNameFromModuleName(AName: string; AProject: IOTAProject = nil): string;
begin
end;

function GetFileNameSearchTypeFromModuleName(AName: string;
  var SearchType: TCnModuleSearchType; AProject: IOTAProject = nil): string;
begin
end;

function CnOtaGetVersionInfoKeys(Project: IOTAProject = nil): TStrings;
begin
end;

procedure GetLibraryPath(Paths: TStrings; IncludeProjectPath: Boolean = True);
begin
end;

function GetComponentUnitName(const ComponentName: string): string;
begin
end;

procedure GetInstalledComponents(Packages, Components: TStrings);
begin
end;

function GetEditControlFromEditorForm(AForm: TCustomForm): TControl;
begin
end;

function GetCurrentEditControl: TControl;
begin
end;

function GetStatusBarFromEditor(EditControl: TControl): TStatusBar;
begin
end;

function GetCurrentSyncButton: TControl;
begin
end;

function GetCurrentSyncButtonVisible: Boolean;
begin
end;

function GetCodeTemplateListBox: TControl;
begin
end;

function GetCodeTemplateListBoxVisible: Boolean;
begin
end;

function IsCurrentEditorInSyncMode: Boolean;
begin
end;

function IsKeyMacroRunning: Boolean;
begin
end;

function GetCurrentCompilingProject: IOTAProject;
begin
end;

function CompileProject(AProject: IOTAProject): Boolean;
begin
end;

{ TCnPaletteWrapper }

procedure TCnPaletteWrapper.BeginUpdate;
begin
end;

constructor TCnPaletteWrapper.Create;
begin
end;

procedure TCnPaletteWrapper.EndUpdate;
begin
end;

function TCnPaletteWrapper.FindTab(const ATab: string): Integer;
begin
end;

function TCnPaletteWrapper.GetActiveTab: string;
begin
end;

function TCnPaletteWrapper.GetEnabled: Boolean;
begin
end;

function TCnPaletteWrapper.GetIsMultiLine: Boolean;
begin
end;

function TCnPaletteWrapper.GetPalToolCount: Integer;
begin
end;

function TCnPaletteWrapper.GetSelectedIndex: Integer;
begin
end;

function TCnPaletteWrapper.GetSelectedToolName: string;
begin
end;

function TCnPaletteWrapper.GetSelector: TSpeedButton;
begin
end;

function TCnPaletteWrapper.GetTabCount: Integer;
begin
end;

function TCnPaletteWrapper.GetTabIndex: Integer;
begin
end;

function TCnPaletteWrapper.GetTabs(Index: Integer): string;
begin
end;

function TCnPaletteWrapper.GetVisible: Boolean;
begin
end;

function TCnPaletteWrapper.SelectComponent(const AComponent,
  ATab: string): Boolean;
begin
end;

procedure TCnPaletteWrapper.SetEnabled(const Value: Boolean);
begin
end;

procedure TCnPaletteWrapper.SetSelectedIndex(const Value: Integer);
begin
end;

procedure TCnPaletteWrapper.SetTabIndex(const Value: Integer);
begin
end;

procedure TCnPaletteWrapper.SetVisible(const Value: Boolean);
begin
end;

function CnPaletteWrapper: TCnPaletteWrapper;
begin
end;
  
{ TCnMessageViewWrapper }

constructor TCnMessageViewWrapper.Create;
begin
end;

procedure TCnMessageViewWrapper.EditMessageSource;
begin
end;

function TCnMessageViewWrapper.GetCurrentMessage: string;
begin
end;

function TCnMessageViewWrapper.GetMessageCount: Integer;
begin
end;

function TCnMessageViewWrapper.GetSelectedIndex: Integer;
begin
end;

function TCnMessageViewWrapper.GetTabCaption: string;
begin
end;

function TCnMessageViewWrapper.GetTabCount: Integer;
begin
end;

function TCnMessageViewWrapper.GetTabIndex: Integer;
begin
end;

function TCnMessageViewWrapper.GetTabSetVisible: Boolean;
begin
end;

procedure TCnMessageViewWrapper.SetSelectedIndex(const Value: Integer);
begin
end;

procedure TCnMessageViewWrapper.SetTabIndex(const Value: Integer);
begin
end;

procedure TCnMessageViewWrapper.UpdateAllItems;
begin
end;

function CnMessageViewWrapper: TCnMessageViewWrapper;
begin
end;

{ TEditorObject }

constructor TEditorObject.Create(AEditControl: TControl;
  AEditView: IOTAEditView);
begin

end;

destructor TEditorObject.Destroy;
begin
  inherited;

end;

function TEditorObject.EditorIsOnTop: Boolean;
begin

end;

function TEditorObject.GetGutterWidth: Integer;
begin

end;

function TEditorObject.GetTopEditor: TControl;
begin

end;

function TEditorObject.GetViewBottomLine: Integer;
begin

end;

function TEditorObject.GetViewLineCount: Integer;
begin

end;

function TEditorObject.GetViewLineNumber(Index: Integer): Integer;
begin

end;

procedure TEditorObject.IDEShowLineNumberChanged;
begin

end;

procedure TEditorObject.SetEditView(AEditView: IOTAEditView);
begin

end;

{ TCnEditControlWrapper }

procedure TCnEditControlWrapper.AddAfterPaintLineNotifier(
  Notifier: TEditorPaintLineNotifier);
begin

end;

procedure TCnEditControlWrapper.AddBeforePaintLineNotifier(
  Notifier: TEditorPaintLineNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditControlNotifier(
  Notifier: TEditorNotifier);
begin

end;

function TCnEditControlWrapper.AddEditor(EditControl: TControl;
  View: IOTAEditView): Integer;
begin

end;

procedure TCnEditControlWrapper.AddEditorChangeNotifier(
  Notifier: TEditorChangeNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditorMouseDownNotifier(
  Notifier: TEditorMouseDownNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditorMouseLeaveNotifier(
  Notifier: TEditorMouseLeaveNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditorMouseMoveNotifier(
  Notifier: TEditorMouseMoveNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditorMouseUpNotifier(
  Notifier: TEditorMouseUpNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditorNcPaintNotifier(
  Notifier: TEditorNcPaintNotifier);
begin

end;

procedure TCnEditControlWrapper.AddEditorVScrollNotifier(
  Notifier: TEditorVScrollNotifier);
begin

end;

procedure TCnEditControlWrapper.AddKeyDownNotifier(
  Notifier: TKeyMessageNotifier);
begin

end;

procedure TCnEditControlWrapper.AddKeyUpNotifier(
  Notifier: TKeyMessageNotifier);
begin

end;

procedure TCnEditControlWrapper.AddNotifier(List: TList;
  Notifier: TMethod);
begin

end;

procedure TCnEditControlWrapper.AfterThemeChange(Sender: TObject);
begin

end;

procedure TCnEditControlWrapper.ApplicationMessage(var Msg: TMsg;
  var Handled: Boolean);
begin

end;

function TCnEditControlWrapper.CalcCharSize: Boolean;
begin

end;

procedure TCnEditControlWrapper.CheckAndSetEditControlMouseHookFlag;
begin

end;

function TCnEditControlWrapper.CheckEditorChanges(
  Editor: TEditorObject): TEditorChangeTypes;
begin

end;

procedure TCnEditControlWrapper.CheckNewEditor(EditControl: TControl;
  View: IOTAEditView);
begin

end;

procedure TCnEditControlWrapper.CheckOptionDlg;
begin

end;

function TCnEditControlWrapper.CheckViewLines(Editor: TEditorObject;
  Context: TCnEditorContext): Boolean;
begin

end;

procedure TCnEditControlWrapper.ClearAndFreeList(var List: TList);
begin

end;

procedure TCnEditControlWrapper.ClearHighlights;
begin

end;

function TCnEditControlWrapper.ClickBreakpointAtActualLine(
  ActualLineNum: Integer; EditControl: TControl): Boolean;
begin

end;

constructor TCnEditControlWrapper.Create(AOwner: TComponent);
begin
  inherited;

end;

procedure TCnEditControlWrapper.DeleteEditor(EditControl: TControl);
begin

end;

destructor TCnEditControlWrapper.Destroy;
begin
  inherited;

end;

procedure TCnEditControlWrapper.DoAfterElide(EditControl: TControl);
begin

end;

procedure TCnEditControlWrapper.DoAfterPaintLine(Editor: TEditorObject;
  LineNum, LogicLineNum: Integer);
begin

end;

procedure TCnEditControlWrapper.DoAfterUnElide(EditControl: TControl);
begin

end;

procedure TCnEditControlWrapper.DoBeforePaintLine(Editor: TEditorObject;
  LineNum, LogicLineNum: Integer);
begin

end;

procedure TCnEditControlWrapper.DoEditControlNotify(EditControl: TControl;
  Operation: TOperation);
begin

end;

procedure TCnEditControlWrapper.DoEditorChange(Editor: TEditorObject;
  ChangeType: TEditorChangeTypes);
begin

end;

procedure TCnEditControlWrapper.DoMouseDown(Editor: TEditorObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer; IsNC: Boolean);
begin

end;

procedure TCnEditControlWrapper.DoMouseLeave(Editor: TEditorObject;
  IsNC: Boolean);
begin

end;

procedure TCnEditControlWrapper.DoMouseMove(Editor: TEditorObject;
  Shift: TShiftState; X, Y: Integer; IsNC: Boolean);
begin

end;

procedure TCnEditControlWrapper.DoMouseUp(Editor: TEditorObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer; IsNC: Boolean);
begin

end;

procedure TCnEditControlWrapper.DoNcPaint(Editor: TEditorObject);
begin

end;

procedure TCnEditControlWrapper.DoVScroll(Editor: TEditorObject);
begin

end;

procedure TCnEditControlWrapper.EditControlProc(EditWindow: TCustomForm;
  EditControl: TControl; Context: Pointer);
begin

end;

procedure TCnEditControlWrapper.EditorRefresh(EditControl: TControl;
  DirtyOnly: Boolean);
begin

end;

procedure TCnEditControlWrapper.ElideLine(EditControl: TControl;
  LineNum: Integer);
begin

end;

procedure TCnEditControlWrapper.GetAttributeAtPos(EditControl: TControl;
  const EdPos: TOTAEditPos; IncludeMargin: Boolean; var Element,
  LineFlag: Integer);
begin

end;

function TCnEditControlWrapper.GetCharHeight: Integer;
begin

end;

function TCnEditControlWrapper.GetCharSize: TSize;
begin

end;

function TCnEditControlWrapper.GetCharWidth: Integer;
begin

end;

function TCnEditControlWrapper.GetEditControl(
  EditView: IOTAEditView): TControl;
begin

end;

function TCnEditControlWrapper.GetEditControlCanvas(
  EditControl: TControl): TCanvas;
begin

end;

function TCnEditControlWrapper.GetEditControlCharHeight(
  EditControl: TControl): Integer;
begin

end;

function TCnEditControlWrapper.GetEditControlInfo(
  EditControl: TControl): TCnEditControlInfo;
begin

end;

function TCnEditControlWrapper.GetEditControlSupportsSyntaxHighlight(
  EditControl: TControl): Boolean;
begin

end;

function TCnEditControlWrapper.GetEditorContext(
  Editor: TEditorObject): TCnEditorContext;
begin

end;

function TCnEditControlWrapper.GetEditorCount: Integer;
begin

end;

function TCnEditControlWrapper.GetEditorObject(
  EditControl: TControl): TEditorObject;
begin

end;

function TCnEditControlWrapper.GetEditors(Index: Integer): TEditorObject;
begin

end;

function TCnEditControlWrapper.GetEditView(
  EditControl: TControl): IOTAEditView;
begin

end;

function TCnEditControlWrapper.GetEditViewFromTabs(
  TabControl: TXTabControl; Index: Integer): IOTAEditView;
begin

end;

function TCnEditControlWrapper.GetFonts(Index: Integer): TFont;
begin

end;

function TCnEditControlWrapper.GetHighlight(
  Index: Integer): THighlightItem;
begin

end;

function TCnEditControlWrapper.GetHighlightCount: Integer;
begin

end;

procedure TCnEditControlWrapper.GetHighlightFromReg;
begin

end;

function TCnEditControlWrapper.GetHighlightName(Index: Integer): string;
begin

end;

function TCnEditControlWrapper.GetLineFromPoint(Point: TPoint;
  EditControl: TControl; EditView: IOTAEditView): Integer;
begin

end;

function TCnEditControlWrapper.GetLineIsElided(EditControl: TControl;
  LineNum: Integer): Boolean;
begin

end;

function TCnEditControlWrapper.GetPointFromEdPos(EditControl: TControl;
  APos: TOTAEditPos): TPoint;
begin

end;

function TCnEditControlWrapper.GetTabWidth: Integer;
begin

end;

function TCnEditControlWrapper.GetTextAtLine(EditControl: TControl;
  LineNum: Integer): string;
begin

end;

function TCnEditControlWrapper.GetTopMostEditControl: TControl;
begin

end;

function TCnEditControlWrapper.GetUseTabKey: Boolean;
begin

end;

function TCnEditControlWrapper.IndexOf(List: TList;
  Notifier: TMethod): Integer;
begin

end;

function TCnEditControlWrapper.IndexOfEditor(
  EditView: IOTAEditView): Integer;
begin

end;

function TCnEditControlWrapper.IndexOfEditor(
  EditControl: TControl): Integer;
begin

end;

function TCnEditControlWrapper.IndexOfHighlight(
  const Name: string): Integer;
begin

end;

function TCnEditControlWrapper.IndexPosToCurPos(EditControl: TControl; Col,
  Line: Integer): Integer;
begin

end;

procedure TCnEditControlWrapper.InitEditControlHook;
begin

end;

procedure TCnEditControlWrapper.LoadFontFromRegistry;
begin

end;

procedure TCnEditControlWrapper.MarkLinesDirty(EditControl: TControl; Line,
  Count: Integer);
begin

end;

procedure TCnEditControlWrapper.Notification(AComponent: TComponent;
  Operation: TOperation);
begin
  inherited;

end;

procedure TCnEditControlWrapper.OnActiveFormChange(Sender: TObject);
begin

end;

procedure TCnEditControlWrapper.OnCallWndProcRet(Handle: HWND;
  Control: TWinControl; Msg: TMessage);
begin

end;

procedure TCnEditControlWrapper.OnGetMsgProc(Handle: HWND;
  Control: TWinControl; Msg: TMessage);
begin

end;

procedure TCnEditControlWrapper.OnIdle(Sender: TObject);
begin

end;

procedure TCnEditControlWrapper.OnSourceEditorNotify(
  SourceEditor: IOTASourceEditor; NotifyType: TCnWizSourceEditorNotifyType;
  EditView: IOTAEditView);
begin

end;

procedure TCnEditControlWrapper.RemoveAfterPaintLineNotifier(
  Notifier: TEditorPaintLineNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveBeforePaintLineNotifier(
  Notifier: TEditorPaintLineNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditControlNotifier(
  Notifier: TEditorNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorChangeNotifier(
  Notifier: TEditorChangeNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorMouseDownNotifier(
  Notifier: TEditorMouseDownNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorMouseLeaveNotifier(
  Notifier: TEditorMouseLeaveNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorMouseMoveNotifier(
  Notifier: TEditorMouseMoveNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorMouseUpNotifier(
  Notifier: TEditorMouseUpNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorNcPaintNotifier(
  Notifier: TEditorNcPaintNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveEditorVScrollNotifier(
  Notifier: TEditorVScrollNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveKeyDownNotifier(
  Notifier: TKeyMessageNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveKeyUpNotifier(
  Notifier: TKeyMessageNotifier);
begin

end;

procedure TCnEditControlWrapper.RemoveNotifier(List: TList;
  Notifier: TMethod);
begin

end;

procedure TCnEditControlWrapper.RepaintEditControls;
begin

end;

procedure TCnEditControlWrapper.ResetFontsFromBasic(ABasicFont: TFont);
begin

end;

procedure TCnEditControlWrapper.ScrollAndClickEditControl(Sender: TObject);
begin

end;

procedure TCnEditControlWrapper.SetFonts(const Index: Integer;
  const Value: TFont);
begin

end;

procedure TCnEditControlWrapper.UnElideLine(EditControl: TControl;
  LineNum: Integer);
begin

end;

function TCnEditControlWrapper.UpdateCharSize: Boolean;
begin

end;

procedure TCnEditControlWrapper.UpdateEditControlList;
begin

end;

end.

