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

unit CnPropertyCompareFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnPack ר�Ұ�
* ��Ԫ���ƣ�������ԶԱȴ��嵥Ԫ
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
* ��    ע��
* ����ƽ̨��Win7 + Delphi 5
* ���ݲ��ԣ�δ����
* �� �� �����ô����е��ַ����ݲ����ϱ��ػ�����ʽ
* �޸ļ�¼��2021.04.18
*               ������Ԫ��ʵ�ֻ�������
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNDESIGNWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs, Contnrs,
  TypInfo, StdCtrls, ComCtrls, ToolWin, Menus, ExtCtrls, ActnList, CommCtrl, Grids,
  {$IFNDEF STAND_ALONE}
  CnWizClasses, CnWizUtils, CnWizIdeUtils, CnWizManager, CnComponentSelector,
  {$ENDIF}
  {$IFDEF SUPPORT_ENHANCED_RTTI} Rtti, {$ENDIF} IniFiles, CnIni,
  CnConsts, CnWizConsts, CnWizMultiLang, CnCommon, CnPropSheetFrm, CnWizShareImages;

const
  WM_SYNC_SELECT = WM_USER + $30;

type
{$IFNDEF STAND_ALONE}
  TCnPropertyCompareManager = class;

  TCnSelectCompareExecutor = class(TCnContextMenuExecutor)
  {* ���һ��ѡ������Ĳ˵����ʾΪѡΪ�����Ƚ����}
  private
    FManager: TCnPropertyCompareManager;
  public
    function GetActive: Boolean; override;
    function GetCaption: string; override;

    property Manager: TCnPropertyCompareManager read FManager write FManager;
  end;

  TCnDoCompareExecutor = class(TCnContextMenuExecutor)
  {* ���һ��������ѡ������Ĳ˵����ʾΪ�� XX �Ƚϣ��������Ƚ�}
  private
    FManager: TCnPropertyCompareManager;
  public
    function GetActive: Boolean; override;
    function GetCaption: string; override;

    property Manager: TCnPropertyCompareManager read FManager write FManager;
  end;

  TCnPropertyCompareManager = class(TComponent)
  private
    FSelectExecutor: TCnSelectCompareExecutor; // ֻѡ��һ��ʱ������ѡΪ���
    FCompareExecutor: TCnDoCompareExecutor;    // ֻѡ����һ��ʱ���� xxxx �Ƚϣ���ѡ������ʱ�Ƚ�����
    FLeftComponent: TComponent;
    FRightObject: TComponent;
    FSelection: TList;
    FSameType: Boolean;
    FIgnoreProperties: TStringList;
    FOnlyShowDiff: Boolean;
    FShowMenu: Boolean;
    FShowEvents: Boolean;
    FGridFont: TFont;
    procedure SetLeftComponent(const Value: TComponent);
    procedure SetRightComponent(const Value: TComponent);
    function GetSelectionCount: Integer;
    procedure SelectExecute(Sender: TObject);
    procedure CompareExecute(Sender: TObject);
    procedure SetIgnoreProperties(const Value: TStringList);
    procedure SetShowMenu(const Value: Boolean);
    procedure SetGridFont(const Value: TFont);
  protected
    procedure Notification(AComponent: TComponent;
      Operation: TOperation); override;

    procedure RegisterMenu;
    procedure UnRegisterMenu;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

    procedure LoadSettings(Ini: TCustomIniFile);
    procedure SaveSettings(Ini: TCustomIniFile);

    property LeftComponent: TComponent read FLeftComponent write SetLeftComponent;
    property RightObject: TComponent read FRightObject write SetRightComponent;
    property SelectionCount: Integer read GetSelectionCount;

    property ShowMenu: Boolean read FShowMenu write SetShowMenu;
    {* �Ƿ������������ʾ�Ƚ���ص��Ҽ��˵�}
    property OnlyShowDiff: Boolean read FOnlyShowDiff write FOnlyShowDiff;
    {* ��ס��ֻ��ʾ��ͬ���Ե�ѡ��}
    property SameType: Boolean read FSameType write FSameType;
    {* �Ƚ�ʱ�Ƿ�Ҫ�����������Ͷ���ͬ������ͬ��������������ͬ����}
    property IgnoreProperties: TStringList read FIgnoreProperties write SetIgnoreProperties;
    {* ȫ����ֵʱҪ���Ե������б��� Name ��}
    property ShowEvents: Boolean read FShowEvents write FShowEvents;
    {* �Ƿ���ʾ�¼�}
    property GridFont: TFont read FGridFont write SetGridFont;
    {* ��ʾ�����壬����Ϊ nil}
  end;

{$ENDIF}

  TCnDiffPropertyObject = class(TCnPropertyObject)
  private
    FIsSingle: Boolean;
    FModified: Boolean;
    FMethod: TMethod;
  public
    property Method: TMethod read FMethod write FMethod;
    {* ������¼��������¼�}
    property IsSingle: Boolean read FIsSingle write FIsSingle;
    {* �Զ��Ƿ������Զ�Ӧ}
    property Modified: Boolean read FModified write FModified;
    {* �Ƿ�Ķ���}
  end;

  TCnPropertyCompareForm = class(TCnTranslateForm)
    mmMain: TMainMenu;
    File1: TMenuItem;
    Exit1: TMenuItem;
    tlbMain: TToolBar;
    btnNewCompare: TToolButton;
    pnlMain: TPanel;
    spl2: TSplitter;
    pnlLeft: TPanel;
    pnlRight: TPanel;
    pnlDisplay: TPanel;
    pbCompare: TPaintBox;
    pbPos: TPaintBox;
    actlstPropertyCompare: TActionList;
    actExit: TAction;
    actSelectLeft: TAction;
    actSelectRight: TAction;
    actPropertyToRight: TAction;
    actPropertyToLeft: TAction;
    pmGrid: TPopupMenu;
    actRefresh: TAction;
    actPrevDiff: TAction;
    actNextDiff: TAction;
    gridLeft: TStringGrid;
    gridRight: TStringGrid;
    actNewCompare: TAction;
    actCompareObjProp: TAction;
    Select1: TMenuItem;
    SelectLeftComponent1: TMenuItem;
    SelectRight1: TMenuItem;
    actNewCompare1: TMenuItem;
    Refresh1: TMenuItem;
    Assign1: TMenuItem;
    actPropertyToLeft1: TMenuItem;
    actPropertyToRight1: TMenuItem;
    PreviousDifferent1: TMenuItem;
    NextDifferent1: TMenuItem;
    Help1: TMenuItem;
    actAllToLeft: TAction;
    actAllToRight: TAction;
    actHelp: TAction;
    Help2: TMenuItem;
    N1: TMenuItem;
    AllToLeft1: TMenuItem;
    AllToRight1: TMenuItem;
    ToLeft1: TMenuItem;
    ToRight1: TMenuItem;
    AllToLeft2: TMenuItem;
    AllToRight2: TMenuItem;
    PreviousDifferent2: TMenuItem;
    NextDifferent2: TMenuItem;
    AllToLeft3: TMenuItem;
    btnRefresh: TToolButton;
    btnSelectLeft: TToolButton;
    btnSelectRight: TToolButton;
    btn1: TToolButton;
    btn2: TToolButton;
    btnPropertyToLeft: TToolButton;
    btnPropertyToRight: TToolButton;
    btnAllToLeft: TToolButton;
    btnAllToRight: TToolButton;
    btn7: TToolButton;
    btnPrevDiff: TToolButton;
    btnNextDiff: TToolButton;
    btn3: TToolButton;
    btnHelp: TToolButton;
    actOptions: TAction;
    N2: TMenuItem;
    Options1: TMenuItem;
    btnOptions: TToolButton;
    btnExit: TToolButton;
    actListLeft: TAction;
    actListRight: TAction;
    SelectLeftComponent2: TMenuItem;
    SelectRightComponent1: TMenuItem;
    N3: TMenuItem;
    btnListLeft: TToolButton;
    btnListRight: TToolButton;
    actOnlyDiff: TAction;
    N4: TMenuItem;
    OnlyShowDifferentProperties1: TMenuItem;
    btnOnlyDiff: TToolButton;
    pnlLeftName: TPanel;
    pnlRightName: TPanel;
    actShowEvents: TAction;
    ShowEvents1: TMenuItem;
    btn4: TToolButton;
    btnShowEvents: TToolButton;
    procedure FormCreate(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure actSelectLeftExecute(Sender: TObject);
    procedure actSelectRightExecute(Sender: TObject);
    procedure pnlResize(Sender: TObject);
    procedure gridDrawCell(Sender: TObject; ACol, ARow: Integer;
      Rect: TRect; State: TGridDrawState);
    procedure gridSelectCell(Sender: TObject; ACol, ARow: Integer;
      var CanSelect: Boolean);
    procedure gridTopLeftChanged(Sender: TObject);
    procedure actRefreshExecute(Sender: TObject);
    procedure actExitExecute(Sender: TObject);
    procedure actNewCompareExecute(Sender: TObject);
    procedure actPropertyToRightExecute(Sender: TObject);
    procedure actPropertyToLeftExecute(Sender: TObject);
    procedure actlstPropertyCompareUpdate(Action: TBasicAction;
      var Handled: Boolean);
    procedure actPrevDiffExecute(Sender: TObject);
    procedure actNextDiffExecute(Sender: TObject);
    procedure gridDblClick(Sender: TObject);
    procedure actCompareObjPropExecute(Sender: TObject);
    procedure actHelpExecute(Sender: TObject);
    procedure actAllToLeftExecute(Sender: TObject);
    procedure actAllToRightExecute(Sender: TObject);
    procedure pbPosPaint(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure pbComparePaint(Sender: TObject);
    procedure pbCompareMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure actOptionsExecute(Sender: TObject);
    procedure actListLeftExecute(Sender: TObject);
    procedure actListRightExecute(Sender: TObject);
    procedure actOnlyDiffExecute(Sender: TObject);
    procedure actShowEventsExecute(Sender: TObject);
  private
    FOnlyDiff: Boolean;
    FShowEvents: Boolean;
    FLeftObject: TObject;
    FRightObject: TObject;
    FLeftProperties: TObjectList;
    FRightProperties: TObjectList;
    FCompareBmp: TBitmap;
    procedure UpdateFont;
{$IFDEF SUPPORT_ENHANCED_RTTI}
    function ListContainsProperty(const APropName: string; List: TObjectList): Boolean;
{$ENDIF}
    function TransferProperty(PFrom, PTo: TCnDiffPropertyObject; FromObj, ToObj: TObject): Boolean;
    {* �����ͬ�����Ը�ֵ�����ظ�ֵ�Ƿ�ɹ�}
    procedure SelectGridRow(Grid: TStringGrid; ARow: Integer);
{$IFDEF SUPPORT_ENHANCED_RTTI}
    procedure LoadOneRttiProp(var AProp: TCnDiffPropertyObject; AObject: TObject;
      RttiProperty: TRttiProperty);
    {* ���� RTTI ��ʽ���ض����һ�����ԣ���� AProp���紫��� AProp Ϊ nil ���ڲ�����}
{$ENDIF}
    procedure LoadOneClassicProp(var AProp: TCnDiffPropertyObject; AObject: TObject; PropInfo: PPropInfo);
    {* �Ծɵķ�ʽ���ض����һ�����ԣ���� AProp���紫��� AProp Ϊ nil ���ڲ�����}
    procedure LoadOneProp(var AProp: TCnDiffPropertyObject; AObject: TObject; const PropName: string);
    {* ��ͳ���ض����һ�����ԣ����������������������}
    procedure LoadProperty(List: TObjectList; AObject: TObject);
    {* �����¡������پɵķ�ʽ����һ���������������}
    procedure MakeAlignList(SameType: Boolean = False);
    {* �������Ի�����룬�м����հ�}
    procedure MakeSingleMarks;
    {* ���������Ա�ע�Է��Ƿ�Ϊ��}
    procedure GetGridSelectObjects(var SelectLeft, SelectRight: Integer;
      var LeftObj, RightObj: TCnDiffPropertyObject);
    procedure UpdateCompareBmp;
    procedure FillGridWithProperties(G: TStringGrid; Props: TObjectList; IsRefresh: Boolean);
    procedure OnSyncSelect(var Msg: TMessage); message WM_SYNC_SELECT;
{$IFNDEF STAND_ALONE}
    function CreateWizardIni: TCustomIniFile;
{$ENDIF}
  protected
    function GetHelpTopic: string; override;
  public
    procedure LoadListProperties;
    {* ��������������������б�����}
    procedure ShowProperties(IsRefresh: Boolean = False);
    {* �������б�����ʾ������������}
    property LeftObject: TObject read FLeftObject write FLeftObject;
    property RightObject: TObject read FRightObject write FRightObject;
  end;

procedure CompareTwoObjects(ALeft: TObject; ARight: TObject);

{$ENDIF CNWIZARDS_CNDESIGNWIZARD}

implementation

{$IFDEF CNWIZARDS_CNDESIGNWIZARD}

{$R *.DFM}

uses
  {$IFDEF DEBUG} CnDebug, {$ENDIF} CnPropertyCompConfigFrm, CnWizOptions, CnGraphUtils
  {$IFNDEF STAND_ALONE}, CnListCompFrm {$ENDIF};

const
  POS_SELECT_COLOR = clNavy;
  POS_SCROLL_COLOR = $00FFC0C0;
  PROP_DIFF_COLOR = $00C0C0FF;
  GUTTER_DIFF_COLOR = $008080FF;
  PROP_SINGLE_COLOR = clWhite;

  PROPNAME_LEFT_MARGIN = 16;
  PROP_NAME_MIN_WIDTH = 60;
  DEF_IGNORE_PROP = 'Name,Left,Top,TabOrder';

  csShowMenu = 'ShowMenu';
  csOnlyShowDiff = 'OnlyShowDiff';
  csSameType = 'SameType';
  csIgnoreProperties = 'IgnoreProperties';
  csShowEvents = 'ShowEvents';
  csGridFont = 'GridFont';

{$IFNDEF STAND_ALONE}
var
  FManager: TCnPropertyCompareManager = nil;
{$ENDIF}

function PropInfoName(PropInfo: PPropInfo): string;
begin
  Result := string(PropInfo^.Name);
end;

procedure CompareTwoObjects(ALeft: TObject; ARight: TObject);
var
  CompareForm: TCnPropertyCompareForm;
begin
  CompareForm := TCnPropertyCompareForm.Create(Application);

{$IFNDEF STAND_ALONE}
  if FManager <> nil then
  begin
    CompareForm.FOnlyDiff := FManager.OnlyShowDiff;
    CompareForm.actOnlyDiff.Checked := FManager.OnlyShowDiff;
    CompareForm.FShowEvents := FManager.ShowEvents;
    CompareForm.actShowEvents.Checked := FManager.ShowEvents;
  end;
{$ENDIF}

  CompareForm.LeftObject := ALeft;
  CompareForm.RightObject := ARight;
  CompareForm.LoadListProperties;
  CompareForm.ShowProperties;
  CompareForm.Show;
end;

procedure DrawTinyDotLine(Canvas: TCanvas; X1, X2, Y1, Y2: Integer);
var
  XStep, YStep, I: Integer;
begin
  with Canvas do
  begin
    if X1 = X2 then
    begin
      YStep := Abs(Y2 - Y1) div 2; // Y �����ܲ�������ֵ
      if Y1 < Y2 then
      begin
        for I := 0 to YStep - 1 do
        begin
          MoveTo(X1, Y1 + (2 * I + 1));
          LineTo(X1, Y1 + (2 * I + 2));
        end;
      end
      else
      begin
        for I := 0 to YStep - 1 do
        begin
          MoveTo(X1, Y1 - (2 * I + 1));
          LineTo(X1, Y1 - (2 * I + 2));
        end;
      end;
    end
    else if Y1 = Y2 then
    begin
      XStep := Abs(X2 - X1) div 2; // X �����ܲ���
      if X1 < X2 then
      begin
        for I := 0 to XStep - 1 do
        begin
          MoveTo(X1 + (2 * I + 1), Y1);
          LineTo(X1 + (2 * I + 2), Y1);
        end;
      end
      else
      begin
        for I := 0 to XStep - 1 do
        begin
          MoveTo(X1 - (2 * I + 1), Y1);
          LineTo(X1 - (2 * I + 2), Y1);
        end;
      end;
    end;
  end;
end;

{$IFNDEF STAND_ALONE}

{ TCnPropertyCompareManager }

procedure TCnPropertyCompareManager.CompareExecute(Sender: TObject);
var
  Comp, Comp2: TComponent;
begin
  if (SelectionCount = 1) and (FLeftComponent <> nil) then
  begin
    Comp := TComponent(FSelection[0]);
    if (Comp <> nil) and (Comp <> FLeftComponent) then
    begin
      RightObject := Comp;
{$IFDEF DEBUG}
      CnDebugger.LogMsg('TCnPropertyCompareManager Compare Execute for Selected and Left.');
{$ENDIF}
    end;
  end
  else if SelectionCount = 2 then
  begin
    Comp := TComponent(FSelection[0]);
    Comp2 := TComponent(FSelection[1]);
    if (Comp <> nil) and (Comp2 <> nil) and (Comp <> Comp2) then
    begin
{$IFDEF DEBUG}
    CnDebugger.LogFmt('TCnPropertyCompareManager Compare Execute for 2 Selected Components: %s vs %s.',
      [Comp.Name, Comp2.Name]);
{$ENDIF}
      LeftComponent := Comp;
      RightObject := Comp2;
    end;
  end;

  CompareTwoObjects(LeftComponent, RightObject);
end;

constructor TCnPropertyCompareManager.Create(AOwner: TComponent);
begin
  inherited;
  FManager := Self;
  FSelection := TList.Create;
  FIgnoreProperties := TStringList.Create;
  // �ӳٴ��� FGridFont
  RegisterMenu;
end;

destructor TCnPropertyCompareManager.Destroy;
begin
  FGridFont.Free;
  FIgnoreProperties.Free;
  FSelection.Free;
  inherited;
end;

function TCnPropertyCompareManager.GetSelectionCount: Integer;
begin
  Result := FSelection.Count;
end;

procedure TCnPropertyCompareManager.LoadSettings(Ini: TCustomIniFile);
var
  Temp1, Temp2: TFont;
begin
  ShowMenu := Ini.ReadBool('', csShowMenu, True);
  FOnlyShowDiff := Ini.ReadBool('', csOnlyShowDiff, FOnlyShowDiff);
  FSameType := Ini.ReadBool('', csSameType, FSameType);
  FIgnoreProperties.CommaText := Ini.ReadString('', csIgnoreProperties, DEF_IGNORE_PROP);
  FShowEvents := Ini.ReadBool('', csShowEvents, False);

  Temp1 := nil;
  Temp2 := nil;

  with TCnIniFile.Create(Ini) do
  try
    Temp1 := TFont.Create;
    Temp2 := TFont.Create;
    Temp2 := ReadFont('', csGridFont, Temp2);

    if FontEqual(Temp1, Temp2) then
    begin
      // Temp2 û�б仯��˵��û�����ã����� FGridFont Ϊ nil
      FreeAndNil(FGridFont);
    end
    else
    begin
      FGridFont := TFont.Create;
      FGridFont.Assign(Temp2);
    end;

    if FGridFont <> nil then
      GridFont := ReadFont('', csGridFont, FGridFont);
  finally
    Temp2.Free;
    Temp1.Free;
    Free;
  end;
end;

procedure TCnPropertyCompareManager.Notification(AComponent: TComponent;
  Operation: TOperation);
begin
  inherited;
  if Operation = opRemove then
  begin
    if AComponent = FLeftComponent then
    begin
      FLeftComponent := nil;
{$IFDEF DEBUG}
      CnDebugger.LogMsg('TCnPropertyCompareManager Get Free Notification. Left set nil.');
{$ENDIF}
    end
    else if AComponent = FRightObject then
    begin
      FRightObject := nil;
{$IFDEF DEBUG}
      CnDebugger.LogMsg('TCnPropertyCompareManager Get Free Notification. Right set nil.');
{$ENDIF}
    end;
  end;
end;

procedure TCnPropertyCompareManager.RegisterMenu;
begin
  if (FSelectExecutor = nil) and (FCompareExecutor = nil) then
  begin
    FSelectExecutor := TCnSelectCompareExecutor.Create;
    FCompareExecutor := TCnDoCompareExecutor.Create;

    FSelectExecutor.Manager := Self;
    FCompareExecutor.Manager := Self;

    FSelectExecutor.OnExecute := SelectExecute;
    FCompareExecutor.OnExecute := CompareExecute;

    RegisterDesignMenuExecutor(FSelectExecutor);
    RegisterDesignMenuExecutor(FCompareExecutor);
  end;
end;

procedure TCnPropertyCompareManager.SaveSettings(Ini: TCustomIniFile);
begin
  Ini.WriteBool('', csShowMenu, FShowMenu);
  Ini.WriteBool('', csOnlyShowDiff, FOnlyShowDiff);
  Ini.WriteBool('', csSameType, FSameType);
  Ini.WriteString('', csIgnoreProperties, FIgnoreProperties.CommaText);
  Ini.WriteBool('', csShowEvents, FShowEvents);

  with TCnIniFile.Create(Ini) do
  try
    if FGridFont <> nil then
      WriteFont('', csGridFont, FGridFont);
  finally
    Free;
  end;
end;

procedure TCnPropertyCompareManager.SelectExecute(Sender: TObject);
var
  Comp: TComponent;
begin
  if SelectionCount = 1 then
  begin
    Comp := TComponent(FSelection[0]);
    if Comp <> nil then
      LeftComponent := Comp;
  end;
end;

procedure TCnPropertyCompareManager.SetGridFont(const Value: TFont);
begin
  if (FGridFont <> nil) and (Value <> nil) then
    FGridFont.Assign(Value);
end;

procedure TCnPropertyCompareManager.SetIgnoreProperties(
  const Value: TStringList);
begin
  FIgnoreProperties.Assign(Value);
end;

procedure TCnPropertyCompareManager.SetLeftComponent(
  const Value: TComponent);
begin
  if FLeftComponent <> Value then
  begin
    if FLeftComponent <> nil then
      FLeftComponent.RemoveFreeNotification(Self);
    FLeftComponent := Value;

{$IFDEF DEBUG}
    if FLeftComponent = nil then
      CnDebugger.LogMsg('TCnPropertyCompareManager LeftComponent Set to nil.')
    else
      CnDebugger.LogMsg('TCnPropertyCompareManager LeftComponent Set to ' + FLeftComponent.Name);
{$ENDIF}

    if FLeftComponent <> nil then
      FLeftComponent.FreeNotification(Self);
  end;
end;

procedure TCnPropertyCompareManager.SetRightComponent(
  const Value: TComponent);
begin
  if FRightObject <> Value then
  begin
    if FRightObject <> nil then
      FRightObject.RemoveFreeNotification(Self);
    FRightObject := Value;

{$IFDEF DEBUG}
    if FRightObject = nil then
      CnDebugger.LogMsg('TCnPropertyCompareManager RightComponent Set to nil.')
    else
      CnDebugger.LogMsg('TCnPropertyCompareManager RightComponent Set to ' + FRightObject.Name);
{$ENDIF}

    if FRightObject <> nil then
      FRightObject.FreeNotification(Self);
  end;
end;

procedure TCnPropertyCompareManager.SetShowMenu(const Value: Boolean);
begin
  if FShowMenu <> Value then
  begin
    FShowMenu := Value;
    if FShowMenu then
      RegisterMenu
    else
      UnRegisterMenu;
  end;
end;

procedure TCnPropertyCompareManager.UnRegisterMenu;
begin
  if (FSelectExecutor <> nil) and (FCompareExecutor <> nil) then
  begin
    UnRegisterDesignMenuExecutor(FSelectExecutor);
    UnRegisterDesignMenuExecutor(FCompareExecutor);

    FSelectExecutor := nil;
    FCompareExecutor := nil;
  end;
end;

{ TCnSelectCompareExecutor }

function TCnSelectCompareExecutor.GetActive: Boolean;
begin
  // ֻѡ��һ��ʱ����
  Result := FManager.SelectionCount = 1;
{$IFDEF DEBUG}
  CnDebugger.LogBoolean(Result, 'TCnSelectCompareExecutor GetActive');
{$ENDIF}
end;

function TCnSelectCompareExecutor.GetCaption: string;
var
  Comp: TComponent;
begin
  Result := '';
  IdeGetFormSelection(FManager.FSelection);

  // ֻѡ��һ��ʱ������Ϊѡ��Ϊ���Ƚ����
  if FManager.SelectionCount = 1 then
  begin
{$IFDEF DEBUG}
    CnDebugger.LogPointer(FManager.FSelection[0], 'TCnSelectCompareExecutor FManager.FSelection[0]');
{$ENDIF}
    Comp := TComponent(FManager.FSelection[0]);
    if Comp <> nil then // ��Щ�������ֻ��������
    begin
      try
        Result := Format(SCnPropertyCompareSelectCaptionFmt, [Comp.Name, Comp.ClassName]);
      except
        Result := Format(SCnPropertyCompareSelectCaptionFmt, ['<unknown>', '<Unknown>']);
      end;
    end;
  end;
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnSelectCompareExecutor GetCaption: ' + Result);
{$ENDIF}
end;

{ TCnDoCompareExecutor }

function TCnDoCompareExecutor.GetActive: Boolean;
begin
  // ֻѡ��һ��ʱ������ Left ʱ������
  // ѡ������ʱ������
  Result := (FManager.SelectionCount = 2) or
    ((FManager.LeftComponent <> nil) and (FManager.SelectionCount = 1));
{$IFDEF DEBUG}
  CnDebugger.LogBoolean(Result, 'TCnDoCompareExecutor GetActive');
{$ENDIF}
end;

function TCnDoCompareExecutor.GetCaption: string;
var
  Comp, Comp2: TComponent;
begin
  Result := '';
  IdeGetFormSelection(FManager.FSelection);
  // ֻѡ��һ��ʱ������ Left ʱ�������� Left �Ƚ�
  // ѡ������ʱ�����رȽ�����

  if FManager.SelectionCount = 1 then
  begin
    Comp := TComponent(FManager.FSelection[0]);
    if (Comp <> nil) and (FManager.LeftComponent <> nil) then
      Result := Format(SCnPropertyCompareToComponentsFmt,
        [FManager.LeftComponent.Name, FManager.LeftComponent.ClassName]);
  end
  else if FManager.SelectionCount = 2 then
  begin
    Comp := TComponent(FManager.FSelection[0]);
    Comp2 := TComponent(FManager.FSelection[1]);
    Result := Format(SCnPropertyCompareTwoComponentsFmt,
      [Comp.Name, Comp.ClassName, Comp2.Name, Comp2.ClassName]);
  end;
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnDoCompareExecutor GetCaption: ' + Result);
{$ENDIF}
end;

{$ENDIF}

function PropertyListCompare(Item1, Item2: Pointer): Integer;
var
  P1, P2: TCnPropertyObject;
begin
  P1 := TCnPropertyObject(Item1);
  P2 := TCnPropertyObject(Item2);

  if (P1.PropType in tkMethods) and not (P2.PropType in tkMethods) then
    Result := 1
  else if (P2.PropType in tkMethods) and not (P1.PropType in tkMethods) then
    Result := -1
  else
    Result := CompareStr(P1.PropName, P2.PropName);
end;

{ TCnPropertyCompareForm }

procedure TCnPropertyCompareForm.LoadListProperties;
var
  I: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  FLeftProperties.Clear;
  FRightProperties.Clear;

  if FLeftObject <> nil then
    LoadProperty(FLeftProperties, FLeftObject);
  if FRightObject <>nil then
    LoadProperty(FRightProperties, FRightObject);

  FLeftProperties.Sort(PropertyListCompare);
  FRightProperties.Sort(PropertyListCompare);

  // �������Զ��룬�Դﵽ���б�����һ��
  MakeAlignList({$IFNDEF STAND_ALONE}FManager.SameType{$ENDIF});
  MakeSingleMarks;

  // ����Ҫ��ɾ����ͬ�ģ��򲻶�Ӧ�ģ�ֻ����ͬ��
  if FOnlyDiff then
  begin
    for I := FLeftProperties.Count - 1 downto 0 do
    begin
      Pl := TCnDiffPropertyObject(FLeftProperties[I]);
      Pr := TCnDiffPropertyObject(FRightProperties[I]);

      if (Pl = nil) or (Pr = nil) then
      begin
        FLeftProperties.Delete(I);
        FRightProperties.Delete(I);
      end
      else if Pl.IsSingle or Pr.IsSingle then
      begin
        FLeftProperties.Delete(I);
        FRightProperties.Delete(I);
      end
      else if Pl.DisplayValue = Pr.DisplayValue then
      begin
        FLeftProperties.Delete(I);
        FRightProperties.Delete(I);
      end;
    end;
  end;
end;

procedure TCnPropertyCompareForm.FormCreate(Sender: TObject);
begin
{$IFNDEF STAND_ALONE}
  WizOptions.ResetToolbarWithLargeIcons(tlbMain);
{$ENDIF}

  FLeftProperties := TObjectList.Create(True);
  FRightProperties := TObjectList.Create(True);

  pnlLeft.OnResize(pnlLeft);
  pnlRight.OnResize(pnlRight);

  FCompareBmp := TBitmap.Create;
  FCompareBmp.Canvas.Brush.Color := clWindow;

  UpdateFont;
end;

{$IFDEF SUPPORT_ENHANCED_RTTI}

procedure TCnPropertyCompareForm.LoadOneRttiProp(var AProp: TCnDiffPropertyObject;
  AObject: TObject; RttiProperty: TRttiProperty);
var
  DataSize: Integer;
begin
  if RttiProperty <> nil then
  begin
    if not (RttiProperty.Visibility in [mvPublished]) then // ע�ⲿ�� published �������������ȡ�� public ���²���ʾ�������� IDE �� Bug���ƹ�������ʹ�þ� RTTI
      Exit;

    if AProp = nil then
      AProp := TCnDiffPropertyObject.Create;
    AProp.IsNewRTTI := True;

    AProp.PropName := RttiProperty.Name;
    AProp.PropType := RttiProperty.PropertyType.TypeKind;
    AProp.IsObjOrIntf := AProp.PropType in [tkClass, tkInterface];

    // ��д��Ȩ�ޣ�����ָ�����ͣ��ſ��޸ģ����������û����
    AProp.CanModify := (RttiProperty.IsWritable) and (RttiProperty.PropertyType.TypeKind
      in CnCanModifyPropTypes);

    if RttiProperty.IsReadable then
    begin
      try
        AProp.PropRttiValue := RttiProperty.GetValue(AObject)
      except
        // Getting Some Property causes Exception. Catch it.
        AProp.PropRttiValue := nil;
      end;

      AProp.ObjValue := nil;
      AProp.IntfValue := nil;
      try
        if AProp.IsObjOrIntf and RttiProperty.GetValue(AObject).IsObject then
          AProp.ObjValue := RttiProperty.GetValue(AObject).AsObject
        else if AProp.IsObjOrIntf and (RttiProperty.GetValue(AObject).TypeInfo <> nil) and
          (RttiProperty.GetValue(AObject).TypeInfo^.Kind = tkInterface) then
          AProp.IntfValue := RttiProperty.GetValue(AObject).AsInterface;
      except
        // Getting Some Property causes Exception. Catch it.;
      end;
    end
    else
      AProp.PropRttiValue := SCnCanNotReadValue;

    if AProp.PropType in tkMethods then
    begin
      AProp.CanModify := True;
      DataSize := RttiProperty.GetValue(AObject).DataSize;
      if DataSize = SizeOf(TMethod) then
      begin
        RttiProperty.GetValue(AObject).ExtractRawData(@AProp.Method);
        if AProp.Method.Data <> nil then
        begin
          try
            AProp.DisplayValue := TObject(AProp.Method.Data).MethodName(AProp.Method.Code);
          except
            ;
          end;
        end;
      end;
    end
    else
      AProp.DisplayValue := GetRttiPropValueStr(AObject, RttiProperty);
  end;
end;

{$ENDIF}

procedure TCnPropertyCompareForm.LoadOneClassicProp(var AProp: TCnDiffPropertyObject;
  AObject: TObject; PropInfo: PPropInfo);
begin
  if AProp = nil then
    AProp := TCnDiffPropertyObject.Create;

  AProp.PropName := PropInfoName(PropInfo);
  AProp.PropType := PropInfo^.PropType^^.Kind;
  AProp.IsObjOrIntf := AProp.PropType in [tkClass, tkInterface];

  // ��д��Ȩ�ޣ�����ָ�����ͣ��ſ��޸ģ����������û����
  AProp.CanModify := (PropInfo^.SetProc <> nil) and (PropInfo^.PropType^^.Kind
    in CnCanModifyPropTypes);

  try
    AProp.PropValue := GetPropValue(AObject, PropInfoName(PropInfo));
  except
    ;
  end;

  AProp.ObjValue := nil;
  AProp.IntfValue := nil;
  if AProp.IsObjOrIntf then
  begin
    if AProp.PropType = tkClass then
      AProp.ObjValue := GetObjectProp(AObject, PropInfo)
    else
      AProp.IntfValue := IUnknown(GetOrdProp(AObject, PropInfo));
  end;

  if AProp.PropType in tkMethods then
  begin
    AProp.CanModify := True;
    AProp.Method := GetMethodProp(AObject, PropInfo);
    if AProp.Method.Data <> nil then
    begin
      try
        AProp.DisplayValue := TObject(AProp.Method.Data).MethodName(AProp.Method.Code);
      except
        ;
      end;
    end;
  end
  else
    AProp.DisplayValue := GetPropValueStr(AObject, PropInfo);
end;

procedure TCnPropertyCompareForm.LoadOneProp(var AProp: TCnDiffPropertyObject;
  AObject: TObject; const PropName: string);
var
{$IFDEF SUPPORT_ENHANCED_RTTI}
  RttiContext: TRttiContext;
  RttiType: TRttiType;
  RttiProperty: TRttiProperty;
{$ENDIF}
  PropInfo: PPropInfo;
begin
  // ���Ծɷ�ʽ�ã���Ϊ LoadOneRttiProp �������Ϊ��������������ò���
  PropInfo := GetPropInfo(AObject, PropName);
  LoadOneClassicProp(AProp, AObject, PropInfo);

{$IFDEF SUPPORT_ENHANCED_RTTI}
  // ��ʵ�·�ʽ���ƺ�Ҳ�ظ��ˣ�ֻ��ָ���õ�ʱ����һ��
  RttiContext := TRttiContext.Create;
  try
    RttiType := RttiContext.GetType(AObject.ClassInfo);
    if RttiType <> nil then
    begin
      RttiProperty := RttiType.GetProperty(PropName);
      LoadOneRttiProp(AProp, AObject, RttiProperty);
    end;
  finally
    RttiContext.Free;
  end;
{$ENDIF}
end;

procedure TCnPropertyCompareForm.LoadProperty(List: TObjectList;
  AObject: TObject);
var
  AProp: TCnDiffPropertyObject;
{$IFDEF SUPPORT_ENHANCED_RTTI}
  RttiContext: TRttiContext;
  RttiType: TRttiType;
  RttiProperty: TRttiProperty;
{$ENDIF}
  PropListPtr: PPropList;
  I, APropCount: Integer;
  PropInfo: PPropInfo;
begin
  // ע��������þɵģ����ֻ���µģ��� RTTI ���е����������� published �Ľ������ public �ĵ��²�����ʾ
  APropCount := GetTypeData(PTypeInfo(AObject.ClassInfo))^.PropCount;
  if APropCount <= 0 then
    Exit;

  GetMem(PropListPtr, APropCount * SizeOf(Pointer));
  try
    GetPropList(PTypeInfo(AObject.ClassInfo), tkAny, PropListPtr);

    for I := 0 to APropCount - 1 do
    begin
      PropInfo := PropListPtr^[I];
      if (PropInfo^.PropType^^.Kind in tkProperties)
        or (FShowEvents and (PropInfo^.PropType^^.Kind in tkMethods)) then
      begin
        AProp := nil;
        LoadOneClassicProp(AProp, AObject, PropInfo);

        if AProp <> nil then
          List.Add(AProp);
      end;
    end;
  finally
    FreeMem(PropListPtr);
  end;

{$IFDEF SUPPORT_ENHANCED_RTTI}
  // D2010 �����ϣ�ʹ���� RTTI ���������ȡ�������ԣ�ʵ�ʿ���Ҳ�಻������Ϊ�ڲ������� published ��
  RttiContext := TRttiContext.Create;
  try
    RttiType := RttiContext.GetType(AObject.ClassInfo);
    if RttiType <> nil then
    begin
      for RttiProperty in RttiType.GetProperties do
      begin
        if RttiProperty.PropertyType.TypeKind in tkProperties then
        begin
          if ListContainsProperty(RttiProperty.Name, List) then // ǰ��ɵġ��Լ����ࡢ�����������ͬ������
            Continue;

          AProp := nil;
          LoadOneRttiProp(AProp, AObject, RttiProperty);
          if AProp <> nil then
            List.Add(AProp);
        end
        else if FShowEvents and (RttiProperty.PropertyType.TypeKind in tkMethods) then
        begin
          if ListContainsProperty(RttiProperty.Name, List) then // ����ɵġ��Լ����ࡢ�����������ͬ������
            Continue;

          AProp := nil;
          LoadOneRttiProp(AProp, AObject, RttiProperty);
          if AProp <> nil then
            List.Add(AProp);
        end;
      end;
    end;
  finally
    RttiContext.Free;
  end;
{$ENDIF}
end;

procedure TCnPropertyCompareForm.FillGridWithProperties(G: TStringGrid;
  Props: TObjectList; IsRefresh: Boolean);
var
  I: Integer;
  P: TCnPropertyObject;
begin
  if (G = nil) or (Props = nil) then
    Exit;

  if not IsRefresh then // ����ʱ��������
  begin
    try
      G.RowCount := 0;
    except
      ;
    end;
  end;

  if G.RowCount <> Props.Count then
  begin
    try
      G.RowCount := Props.Count;
    except
      ;
    end;
  end;

  for I := 0 to Props.Count - 1 do
  begin
    P := TCnPropertyObject(Props[I]);
    if P <> nil then
    begin
      if IsRefresh then
      begin
        if G.Cells[0, I] <> P.PropName then
          G.Cells[0, I] := P.PropName;
        if G.Cells[1, I] <> P.DisplayValue then
          G.Cells[1, I] := P.DisplayValue;
      end
      else
      begin
        G.Cells[0, I] := P.PropName;
        G.Cells[1, I] := P.DisplayValue;
      end;
    end
    else
    begin
      G.Cells[0, I] := '';
      G.Cells[1, I] := '';
    end;
  end;
end;

procedure TCnPropertyCompareForm.ShowProperties(IsRefresh: Boolean);
begin
  pnlLeft.Caption := '';
  pnlRight.Caption := '';

  if FLeftObject <> nil then
  begin
    if LeftObject is TComponent then
      pnlLeftName.Caption := Format('%s: %s', [(LeftObject as TComponent).Name, LeftObject.ClassName])
    else
      pnlLeftName.Caption := Format('$%p: %s', [Pointer(LeftObject), LeftObject.ClassName]);
  end;

  if FRightObject <> nil then
  begin
    if RightObject is TComponent then
      pnlRightName.Caption := Format('%s: %s', [(RightObject as TComponent).Name, RightObject.ClassName])
    else
      pnlRightName.Caption := Format('$%p: %s', [Pointer(RightObject), RightObject.ClassName]);
  end;

  FillGridWithProperties(gridLeft, FLeftProperties, IsRefresh);
  FillGridWithProperties(gridRight, FRightProperties, IsRefresh);

  gridLeft.Invalidate;
  gridRight.Invalidate;
  UpdateCompareBmp;
end;

procedure TCnPropertyCompareForm.MakeAlignList(SameType: Boolean);
var
  L, R, C: Integer;
  PL, PR: TCnPropertyObject;
  Merge: TStringList;
begin
  Merge := TStringList.Create;
  Merge.Duplicates := dupIgnore;

  try
    L := 0;
    R := 0;
    while (L < FLeftProperties.Count) and (R < FRightProperties.Count) do
    begin
      PL := TCnPropertyObject(FLeftProperties[L]);
      PR := TCnPropertyObject(FRightProperties[R]);

      // C := CompareStr(PL.PropName, PR.PropName);
      // ֻ�д������԰���ĸ����ʱ������ CompareStr�����¼�ʱ��д�ɺ��������һ��
      C := PropertyListCompare(PL, PR);
      if C = 0 then
      begin
        // �����Ҫ������������������ͬ�����Ͳ�ͬ���������� C
        if SameType and (PL.PropType <> PR.PropType) then
          C := Ord(PL.PropType) - Ord(PR.PropType);
      end;

      // ���
      if C = 0 then
      begin
        Inc(L);
        Inc(R);

        if SameType then
          Merge.Add(PL.PropName + IntToStr(Ord(PL.PropType)))
        else
          Merge.Add(PL.PropName);
      end
      else if C < 0 then // �����С
      begin
        if SameType then
          Merge.Add(PL.PropName + IntToStr(Ord(PL.PropType)))
        else
          Merge.Add(PL.PropName);
        Inc(L);
      end
      else if C > 0 then // �ұ���С
      begin
        if SameType then
          Merge.Add(PR.PropName + IntToStr(Ord(PR.PropType)))
        else
          Merge.Add(PR.PropName);
        Inc(R);
      end;
    end;

    // Merge �еõ��鲢�������㣬Ȼ�����Ҹ����Լ�ÿһ���Ӧ������
    L := 0;
    while L < FLeftProperties.Count do
    begin
      PL := TCnPropertyObject(FLeftProperties[L]);
      if SameType then
        R := Merge.IndexOf(PL.PropName + IntToStr(Ord(PL.PropType)))
      else
        R := Merge.IndexOf(PL.PropName);

      // R һ���� >= L
      if R > L then
      begin
        // �� L ��ǰһ�������ʵ������� nil
        for C := 1 to R - L do
          FLeftProperties.Insert(L, nil);
        Inc(L, R - L);
      end;

      Inc(L);
    end;

    R := 0;
    while R < FRightProperties.Count do
    begin
      PR := TCnPropertyObject(FRightProperties[R]);
      if SameType then
        L := Merge.IndexOf(PR.PropName + IntToStr(Ord(PR.PropType)))
      else
        L := Merge.IndexOf(PR.PropName);

      // L һ���� >= R
      if L > R then
      begin
        // �� R ��ǰһ�������ʵ������� nil
        for C := 1 to L - R do
          FRightProperties.Insert(R, nil);
        Inc(R, L - R);
      end;

      Inc(R);
    end;

    // β�����ȵĻ�������
    if FLeftProperties.Count > FRightProperties.Count then
    begin
      for L := 0 to FLeftProperties.Count - FRightProperties.Count - 1 do
        FRightProperties.Add(nil);
    end
    else if FRightProperties.Count > FLeftProperties.Count then
    begin
      for L := 0 to FRightProperties.Count - FLeftProperties.Count - 1 do
        FLeftProperties.Add(nil);
    end;
  finally
    Merge.Free;
  end;
end;

procedure TCnPropertyCompareForm.FormResize(Sender: TObject);
begin
  pnlLeft.Width := pnlLeft.Parent.Width div 2 - 5 - pnlDisplay.Width;
end;

procedure TCnPropertyCompareForm.OnSyncSelect(var Msg: TMessage);
var
  Old: TSelectCellEvent;
  G: TStringGrid;
  R: Integer;
  CR: TGridRect;
begin
  if Msg.Msg = WM_SYNC_SELECT then
  begin
    G := TStringGrid(Msg.WParam);
    R := Msg.LParam;

    if G <> nil then
    begin
      CR := G.Selection;
      if (CR.Top <> R) or (CR.Bottom <> R) then
      begin
        Old := G.OnSelectCell;
        G.OnSelectCell := nil;
        if G.Cells[0, R] = '' then // Ŀ����û����
        begin
          CR.Top := -1;
          CR.Bottom := -1;
        end
        else
        begin
          CR.Top := R;
          CR.Bottom := R;
        end;
        CR.Left := 0;
        CR.Right := 1;
        G.Selection := CR;
        G.Invalidate;

        G.OnSelectCell := Old;
      end;
    end;
  end;
end;

procedure TCnPropertyCompareForm.MakeSingleMarks;
var
  I: Integer;
  PL, PR: TCnDiffPropertyObject;
begin
  if FLeftProperties.Count = FRightProperties.Count then
  begin
    for I := 0 to FLeftProperties.Count - 1 do
    begin
      PL := TCnDiffPropertyObject(FLeftProperties[I]);
      PR := TCnDiffPropertyObject(FRightProperties[I]);

      if (PL = nil) and (PR <> nil) then
        PR.IsSingle := True;

      if (PR = nil) and (PL <> nil) then
        PL.IsSingle := True;
    end;
  end;
end;

{$IFDEF SUPPORT_ENHANCED_RTTI}

function TCnPropertyCompareForm.ListContainsProperty(
  const APropName: string; List: TObjectList): Boolean;
var
  I: Integer;
  P: TCnPropertyObject;
begin
  Result := False;
  for I := 0 to List.Count - 1 do
  begin
    P := TCnPropertyObject(List[I]);
    if (P <> nil) and (P.PropName = APropName) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

{$ENDIF}

procedure TCnPropertyCompareForm.actSelectLeftExecute(Sender: TObject);
{$IFNDEF STAND_ALONE}
var
  List: TComponentList;
{$ENDIF}
begin
{$IFNDEF STAND_ALONE}
  List := TComponentList.Create(False);
  try
    if SelectComponentsWithSelector(List) then
    begin
      if List.Count = 1 then
        LeftObject := List[0]
      else if List.Count > 1 then
      begin
        LeftObject := List[0];   // ѡ�����������ϣ��������
        RightObject := List[1];
      end
      else
        Exit;

      LoadListProperties;
      ShowProperties;
    end;
  finally
    List.Free;
  end;
{$ENDIF}
end;

procedure TCnPropertyCompareForm.actSelectRightExecute(Sender: TObject);
{$IFNDEF STAND_ALONE}
var
  List: TComponentList;
{$ENDIF}
begin
{$IFNDEF STAND_ALONE}
  List := TComponentList.Create(False);
  try
    if SelectComponentsWithSelector(List) then
    begin
      if List.Count = 1 then
        RightObject := List[0]
      else if List.Count > 1 then
      begin
        RightObject := List[0];  // ѡ�����������ϣ����Һ���
        LeftObject := List[1];
      end
      else
        Exit;

      LoadListProperties;
      ShowProperties;
    end;
  finally
    List.Free;
  end;
{$ENDIF}
end;

procedure TCnPropertyCompareForm.pnlResize(Sender: TObject);
var
  P: TPanel;
  G: TStringGrid;
  I: Integer;
  C: TControl;
begin
  if Sender is TPanel then
  begin
    P := Sender as TPanel;
    G := nil;
    for I := 0 to P.ControlCount - 1 do
    begin
      C := P.Controls[I];
      if C is TStringGrid then
      begin
        G := C as TStringGrid;
        Break;
      end;
    end;

    if G <> nil then
    begin
      I := (P.Width - 2) div 3;
      if I < PROP_NAME_MIN_WIDTH then
        I := PROP_NAME_MIN_WIDTH;

      G.ColWidths[0] := I;
      G.ColWidths[1] := P.Width - I - 2;
    end;
  end;
end;

procedure TCnPropertyCompareForm.gridDrawCell(Sender: TObject; ACol,
  ARow: Integer; Rect: TRect; State: TGridDrawState);
var
  S: string;
  H, W: Integer;
  G: TStringGrid;
  One, Another: TObjectList;
  P1, P2: TCnDiffPropertyObject;
begin
  G := Sender as TStringGrid;
  if G = gridLeft then
  begin
    One := FLeftProperties;
    Another := FRightProperties;
  end
  else
  begin
    One := FRightProperties;
    Another := FLeftProperties;
  end;

  if ARow >= One.Count then
    P1 := nil
  else
    P1 := TCnDiffPropertyObject(One[ARow]);

  if ARow >= Another.Count then
    P2 := nil
  else
    P2 := TCnDiffPropertyObject(Another[ARow]);

  // ������
  G.Canvas.Font.Assign(G.Font);
  G.Canvas.Font.Color := clBtnText;
  G.Canvas.Brush.Style := bsSolid;

  if ACol = 0 then
  begin
    if (P2 <> nil) and P2.IsSingle then // �Լ�û�жԷ��У��׵ף�
      G.Canvas.Brush.Color := clWhite
    else
      G.Canvas.Brush.Color := clBtnFace;
  end
  else if gdSelected in State then
  begin
    if (P2 <> nil) and P2.IsSingle then // �Լ�û�жԷ��У��׵ף�
    begin
      G.Canvas.Brush.Color := PROP_SINGLE_COLOR;
    end
    else
    begin
      G.Canvas.Brush.Color := clHighlight;
      G.Canvas.Font.Color := clHighlightText;
    end;
  end
  else
  begin
    // ���ݶԱȽ�����ñ���ɫ
    G.Canvas.Brush.Color := clBtnFace;

    if (P1 <> nil) and P1.IsSingle then // �Լ��жԷ�û�У���ͨ�ҵף�
    begin

    end
    else if (P2 <> nil) and P2.IsSingle then // �Լ�û�жԷ��У��׵ף�
    begin
      G.Canvas.Brush.Color := PROP_SINGLE_COLOR;
    end
    else if (P1 <> nil) and (P2 <> nil) then
    begin
      if P1.DisplayValue <> P2.DisplayValue then  // �����Ҳ�ͬ������ף�
        G.Canvas.Brush.Color := PROP_DIFF_COLOR;
    end;
    // ��������ͬ����ͨ�ҵף�
  end;

  G.Canvas.FillRect(Rect);

  // ������
  S := G.Cells[ACol, ARow];

  G.Canvas.Brush.Style := bsClear;
  H := G.Canvas.TextHeight(S);
  H := (Rect.Bottom - Rect.Top - H) div 2;
  if H < 0 then
    H := 0;
  if ACol = 0 then
    W := PROPNAME_LEFT_MARGIN
  else
    W := PROPNAME_LEFT_MARGIN div 2;
  G.Canvas.TextOut(Rect.Left + W, Rect.Top + H, S);

  // ����ָ���
  G.Canvas.Pen.Color := clBtnText;
  G.Canvas.Pen.Style := psSolid;

  DrawTinyDotLine(G.Canvas, Rect.Left, Rect.Right, Rect.Bottom - 1, Rect.Bottom - 1);

  // �� 0 �� 1 ֮�������
  if ACol = 0 then
  begin
    H := Rect.Right - 1;

    G.Canvas.Pen.Color := clBlack;
    G.Canvas.MoveTo(H, Rect.Top);
    G.Canvas.LineTo(H, Rect.Bottom);
  end
  else if ACol = 1 then
  begin
    G.Canvas.Pen.Color := clWhite;
    G.Canvas.MoveTo(Rect.Left, Rect.Top);
    G.Canvas.LineTo(Rect.Left, Rect.Bottom);
    G.Canvas.Pen.Color := clBlack;
    DrawTinyDotLine(G.Canvas, Rect.Right - 1, Rect.Right - 1, Rect.Top, Rect.Bottom);
  end;
end;

procedure TCnPropertyCompareForm.gridSelectCell(Sender: TObject; ACol,
  ARow: Integer; var CanSelect: Boolean);
var
  G: TStringGrid;
begin
  if Sender = gridLeft then
    G := gridRight
  else
    G := gridLeft;

{$IFDEF WIN64}
  PostMessage(Handle, WM_SYNC_SELECT, NativeUInt(G), ARow);
{$ELSE}
  PostMessage(Handle, WM_SYNC_SELECT, Integer(G), ARow);
{$ENDIF}
  pbPos.Invalidate;
end;

procedure TCnPropertyCompareForm.gridTopLeftChanged(Sender: TObject);
var
  G: TStringGrid;
begin
  if Sender = gridLeft then
    G := gridRight
  else
    G := gridLeft;

  G.TopRow := (Sender as TStringGrid).TopRow;
  pbPos.Invalidate;
end;

procedure TCnPropertyCompareForm.actRefreshExecute(Sender: TObject);
begin
  actOnlyDiff.Checked := FOnlyDiff;

  LoadListProperties;
  ShowProperties(True);
end;

procedure TCnPropertyCompareForm.actExitExecute(Sender: TObject);
begin
  Close;
end;

procedure TCnPropertyCompareForm.actNewCompareExecute(Sender: TObject);
var
  CompareForm: TCnPropertyCompareForm;
begin
  CompareForm := TCnPropertyCompareForm.Create(Application);;
  CompareForm.LoadListProperties;
  CompareForm.ShowProperties;
  CompareForm.Show;
end;

procedure TCnPropertyCompareForm.actPropertyToLeftExecute(Sender: TObject);
var
  ARow: Integer;
  POne, PAnother: TCnDiffPropertyObject;
begin
  ARow := gridRight.Selection.Top;
  if (ARow < 0) or (ARow >= FRightProperties.Count) then
    Exit;

  POne := TCnDiffPropertyObject(FRightProperties[ARow]);
  PAnother := TCnDiffPropertyObject(FLeftProperties[ARow]);

  if TransferProperty(POne, PAnother, FRightObject, FLeftObject) then
  begin
    // ֻ���� PAnother ��
    LoadOneProp(PAnother, FLeftObject, PAnother.PropName);
    ShowProperties(True);
    SelectGridRow(gridRight, ARow);
  end;
end;

procedure TCnPropertyCompareForm.actPropertyToRightExecute(
  Sender: TObject);
var
  ARow: Integer;
  POne, PAnother: TCnDiffPropertyObject;
begin
  ARow := gridLeft.Selection.Top;
  if (ARow < 0) or (ARow >= FLeftProperties.Count) then
    Exit;

  POne := TCnDiffPropertyObject(FLeftProperties[ARow]);
  PAnother := TCnDiffPropertyObject(FRightProperties[ARow]);

  if TransferProperty(POne, PAnother, FLeftObject, FRightObject) then
  begin
    // ֻ���� PAnother ��
    LoadOneProp(PAnother, FRightObject, PAnother.PropName);
    ShowProperties(True);
    SelectGridRow(gridLeft, ARow);
  end;
end;

function TCnPropertyCompareForm.TransferProperty(PFrom,
  PTo: TCnDiffPropertyObject; FromObj, ToObj: TObject): Boolean;
var
  V: Variant;
  Obj: TObject;
  M: TMethod;
{$IFDEF SUPPORT_ENHANCED_RTTI}
  RttiContext: TRttiContext;
  RttiTypeFrom, RttiTypeTo: TRttiType;
  RttiPropertyFrom, RttiPropertyTo: TRttiProperty;
{$ENDIF}
begin
  Result := False;
  if (PFrom = nil) or (PTo = nil) or (FromObj = nil) or (ToObj = nil) then
    Exit;

  if (PFrom.PropName <> PTo.PropName) or not PTo.CanModify then
    Exit;

{$IFDEF SUPPORT_ENHANCED_RTTI}

  if PFrom.IsNewRTTI and PTo.IsNewRTTI then
  begin
    RttiContext := TRttiContext.Create;
    try
      RttiTypeFrom := RttiContext.GetType(FromObj.ClassInfo);
      RttiTypeTo := RttiContext.GetType(ToObj.ClassInfo);

      if (RttiTypeFrom = nil) or (RttiTypeTo = nil) then
        Exit;

      RttiPropertyFrom := RttiTypeFrom.GetProperty(PFrom.PropName);
      RttiPropertyTo := RttiTypeTo.GetProperty(PTo.PropName);

      if (RttiPropertyFrom = nil) or (RttiPropertyTo = nil) then
        Exit;

      // ֱ��ͨ�� TValue ��ֵ
      RttiPropertyTo.SetValue(ToObj, RttiPropertyFrom.GetValue(FromObj));
      Result := True;
    finally
      RttiContext.Free;
    end;

    Exit;
  end;

{$ENDIF}

  // Object ��������
  if PFrom.PropType <> PTo.PropType then // ���Ͳ����ֻ���� Variant ǿ�д���
  begin
    V := GetPropValue(FromObj, PFrom.PropName);
    SetPropValue(ToObj, PTo.PropName, V);
  end
  else if PFrom.PropType in [tkClass, tkInterface] then
  begin
    Obj := GetObjectProp(FromObj, PFrom.PropName);
    SetObjectProp(ToObj, PTo.PropName, Obj);
  end
  else if PFrom.PropType = tkMethod then
  begin
    // �õ��¼��ٸ�ֵ��ȥ
    M := GetMethodProp(FromObj, PFrom.PropName);
    SetMethodProp(ToObj, PTo.PropName, M);
  end
  else // ������ͬ������Ҳ�� Variant ����
  begin
    V := GetPropValue(FromObj, PFrom.PropName);
    SetPropValue(ToObj, PTo.PropName, V);
  end;
  Result := True;
end;

procedure TCnPropertyCompareForm.SelectGridRow(Grid: TStringGrid;
  ARow: Integer);
var
  Sel: TGridRect;
begin
  Sel.Top := ARow;
  Sel.Bottom := ARow;
  Sel.Left := 0;
  Sel.Right := Grid.ColCount - 1;

  Grid.Selection := Sel;

  // ���� ARow �ɼ�
  if ARow < Grid.TopRow then
    Grid.TopRow := ARow
  else if ARow > (Grid.TopRow + Grid.VisibleRowCount - 1) then
    Grid.TopRow := ARow - Grid.VisibleRowCount + 1;
end;

procedure TCnPropertyCompareForm.actlstPropertyCompareUpdate(
  Action: TBasicAction; var Handled: Boolean);
var
  Sl, Sr: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  GetGridSelectObjects(Sl, Sr, Pl, Pr);

  if Action = actPropertyToLeft then
    (Action as TCustomAction).Enabled := (Pr <> nil) and not Pr.IsSingle
      and (Pl <> nil) and Pl.CanModify and (Pl.DisplayValue <> Pr.DisplayValue)
  else if Action = actPropertyToRight then
    (Action as TCustomAction).Enabled := (Pl <> nil) and not Pl.IsSingle
      and (Pr <> nil) and Pr.CanModify and (Pl.DisplayValue <> Pr.DisplayValue)
  else if Action = actCompareObjProp then
    (Action as TCustomAction).Enabled := (Pl <> nil) and Pl.IsObjOrIntf
     and (Pr <> nil) and Pr.IsObjOrIntf and ((Pl.ObjValue <> nil) or (Pr.ObjValue <> nil))
  else if (Action = actAllToLeft) or (Action = actAllToRight) or (Action = actPrevDiff)
    or (Action = actNextDiff) then
    (Action as TCustomAction).Enabled := (Pl <> nil) and (Pr <> nil);
end;

procedure TCnPropertyCompareForm.actPrevDiffExecute(Sender: TObject);
var
  I, Sl, Sr: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  GetGridSelectObjects(Sl, Sr, Pl, Pr);

  if (Sl > 0) and (Sr > 0) then
  begin
    for I := Sl - 1 downto 0 do
    begin
      Pl := TCnDiffPropertyObject(FLeftProperties[I]);
      Pr := TCnDiffPropertyObject(FRightProperties[I]);
      if (Pl <> nil) and (Pr <> nil) then
      begin
        if Pl.DisplayValue <> Pr.DisplayValue then
        begin
          SelectGridRow(gridLeft, I);
          SelectGridRow(gridRight, I);
          Exit;
        end;
      end;
    end;
  end;

  ErrorDlg(SCnPropertyCompareNoPrevDiff);
end;

procedure TCnPropertyCompareForm.GetGridSelectObjects(var SelectLeft,
  SelectRight: Integer; var LeftObj, RightObj: TCnDiffPropertyObject);
begin
  SelectLeft := gridLeft.Selection.Top;
  SelectRight := gridRight.Selection.Top;

  if (SelectLeft >= 0) and (SelectLeft < FLeftProperties.Count) then
    LeftObj := TCnDiffPropertyObject(FLeftProperties[SelectLeft])
  else
    LeftObj := nil;

  if (SelectRight >= 0) and (SelectRight < FRightProperties.Count) then
    RightObj := TCnDiffPropertyObject(FRightProperties[SelectRight])
  else
    RightObj := nil;
end;

procedure TCnPropertyCompareForm.actNextDiffExecute(Sender: TObject);
var
  I, Sl, Sr: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  GetGridSelectObjects(Sl, Sr, Pl, Pr);

  if (Sl < FLeftProperties.Count) and (Sr < FRightProperties.Count) then
  begin
    for I := Sl + 1 to FLeftProperties.Count - 1 do
    begin
      Pl := TCnDiffPropertyObject(FLeftProperties[I]);
      Pr := TCnDiffPropertyObject(FRightProperties[I]);
      if (Pl <> nil) and (Pr <> nil) then
      begin
        if Pl.DisplayValue <> Pr.DisplayValue then
        begin
          SelectGridRow(gridLeft, I);
          SelectGridRow(gridRight, I);
          Exit;
        end;
      end;
    end;
  end;

  ErrorDlg(SCnPropertyCompareNoNextDiff);
end;

procedure TCnPropertyCompareForm.gridDblClick(Sender: TObject);
begin
  actCompareObjProp.Execute;
end;

procedure TCnPropertyCompareForm.actCompareObjPropExecute(Sender: TObject);
var
  Sl, Sr: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  GetGridSelectObjects(Sl, Sr, Pl, Pr);
  if (Pl <> nil) and (Pr <> nil) then
  begin
    if Pl.IsObjOrIntf and Pr.IsObjOrIntf then
      CompareTwoObjects(Pl.ObjValue, Pr.ObjValue);
  end;
end;

procedure TCnPropertyCompareForm.actHelpExecute(Sender: TObject);
begin
  ShowFormHelp;
end;

function TCnPropertyCompareForm.GetHelpTopic: string;
begin
  Result := 'CnAlignSizeConfig';
end;

procedure TCnPropertyCompareForm.actAllToLeftExecute(Sender: TObject);
var
  I: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  try
    for I := 0 to FRightProperties.Count - 1 do
    begin
      Pl := TCnDiffPropertyObject(FLeftProperties[I]);
      Pr := TCnDiffPropertyObject(FRightProperties[I]);

      if (Pl = nil) or (Pr = nil) then
        Continue;
      if Pl.IsSingle or Pr.IsSingle then
        Continue;

      if Pl.DisplayValue = Pr.DisplayValue then
        Continue;

{$IFNDEF STAND_ALONE}
      if FManager.IgnoreProperties.IndexOf(Pr.PropName) >= 0 then
        Continue;
{$ENDIF}

      TransferProperty(Pr, Pl, FRightObject, FLeftObject);
    end;
  finally
    LoadListProperties;
    ShowProperties(True);
  end;
end;

procedure TCnPropertyCompareForm.actAllToRightExecute(Sender: TObject);
var
  I: Integer;
  Pl, Pr: TCnDiffPropertyObject;
begin
  try
    for I := 0 to FRightProperties.Count - 1 do
    begin
      Pl := TCnDiffPropertyObject(FLeftProperties[I]);
      Pr := TCnDiffPropertyObject(FRightProperties[I]);

      if (Pl = nil) or (Pr = nil) then
        Continue;
      if Pl.IsSingle or Pr.IsSingle then
        Continue;

      if Pl.DisplayValue = Pr.DisplayValue then
        Continue;

{$IFNDEF STAND_ALONE}
      if FManager.IgnoreProperties.IndexOf(Pl.PropName) >= 0 then
        Continue;
{$ENDIF}

      TransferProperty(Pl, Pr, FLeftObject, FRightObject);
    end;
  finally
    LoadListProperties;
    ShowProperties(True);
  end;
end;

procedure TCnPropertyCompareForm.pbPosPaint(Sender: TObject);
var
  Y1, Y2: Integer;
begin
  with pbPos do
  begin
    Canvas.Brush.Color := clWindow;
    Canvas.FillRect(ClientRect);
    if (gridLeft.RowCount = 0) or (gridRight.RowCount = 0) then
      Exit;

    Y1 := gridLeft.TopRow;
    Y1 := ClientHeight * Y1 div gridLeft.RowCount;

    Y2 := gridLeft.TopRow + gridLeft.VisibleRowCount;
    Y2 := ClientHeight * Y2 div gridLeft.RowCount;

    Canvas.Brush.Color := POS_SCROLL_COLOR;
    Canvas.FillRect(Rect(0, Y1, ClientWidth, Y2));

    Y1 := gridLeft.Selection.Top;
    if Y1 < 0 then
      Y1 := gridRight.Selection.Top;

    Y1 := ClientHeight * Y1 div gridLeft.RowCount + 3;

    Canvas.Pen.Color := POS_SELECT_COLOR;
    Canvas.Pen.Width := 3;
    Canvas.MoveTo(0, Y1);
    Canvas.LineTo(ClientWidth, Y1);
  end;
end;

procedure TCnPropertyCompareForm.FormDestroy(Sender: TObject);
begin
  FRightProperties.Free;
  FLeftProperties.Free;
  FCompareBmp.Free;
end;

procedure TCnPropertyCompareForm.UpdateCompareBmp;
var
  I, J, Y1, Y2: Integer;
  AColor: TColor;
  HeightRatio: single;

  function GetPaintColor(ARow: Integer): TColor;
  var
    Pl, Pr: TCnDiffPropertyObject;
  begin
    // ���ݵ� ARow �еıȽϽ����������ɫ������ͬ����ͬ��ȱʧ���֣���ͬ�����ػ�
    Result := clNone;

    if (ARow < 0) or (ARow >= FLeftProperties.Count) then
      Exit;
    if (ARow < 0) or (ARow >= FRightProperties.Count) then
      Exit;

    Pl := TCnDiffPropertyObject(FLeftProperties[ARow]);
    Pr := TCnDiffPropertyObject(FRightProperties[ARow]);

    if (Pl = nil) or (Pr = nil) then
      Result := PROP_SINGLE_COLOR
    else if Pl.IsSingle or Pr.IsSingle then
      Result := PROP_SINGLE_COLOR
    else if Pl.DisplayValue <> Pr.DisplayValue then
      Result := GUTTER_DIFF_COLOR;
  end;

begin
  if (gridLeft.RowCount = 0) or (gridRight.RowCount = 0) then
    Exit;

  HeightRatio := Screen.Height / gridLeft.RowCount;

  FCompareBmp.Height := Screen.Height;
  FCompareBmp.Width := pbCompare.ClientWidth;
  FCompareBmp.Canvas.Pen.Width := 2;
  FCompareBmp.Canvas.Brush.Color := clBtnFace;
  FCompareBmp.Canvas.FillRect(Rect(0, 0, FCompareBmp.Width, FCompareBmp.Height));

  I := 0;
  while I < gridLeft.RowCount do
  begin
    AColor := GetPaintColor(I);

    if AColor = clNone then
      Inc(I)
    else
    begin
      J := I + 1;
      while (J < gridLeft.RowCount) and (GetPaintColor(J) = AColor) do
        Inc(J);

      FCompareBmp.Canvas.Brush.Color := AColor;
      Y1 := Trunc(I * HeightRatio);
      Y2 := Trunc(J * HeightRatio);
      FCompareBmp.Canvas.FillRect(Rect(0, Y1, FCompareBmp.Width, Y2));
      I := J;
    end;
  end;
  pbCompare.Invalidate;
end;

procedure TCnPropertyCompareForm.pbComparePaint(Sender: TObject);
begin
  with pbCompare do
    Canvas.StretchDraw(Rect(0, 0, Width, Height), FCompareBmp);
end;

procedure TCnPropertyCompareForm.pbCompareMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
const
  GAP = 4;
var
  R: Integer;
begin
  if Button = mbLeft then
  begin
    R := Trunc(Y / pbCompare.Height *
      (gridLeft.RowCount - 1)) - gridLeft.ClientHeight div
      gridLeft.DefaultRowHeight div 2;

    if R < 0 then
      R := 0;

    gridLeft.TopRow := R;

    // Sel: Y: 4 -> 0, ClientHeight - 4 -> RowCount - 1
    Y := Y - GAP;
    if Y < 0 then
      Y := 0;

    R := Trunc(Y * gridLeft.RowCount / (pbCompare.Height - GAP * 2));
    if R >= gridLeft.RowCount then
      R := gridLeft.RowCount - 1;

    SelectGridRow(gridLeft, R);
    SelectGridRow(gridRight, R);
  end;
end;

procedure TCnPropertyCompareForm.actOptionsExecute(Sender: TObject);
begin
  with TCnPropertyCompConfigForm.Create(nil) do
  begin
{$IFNDEF STAND_ALONE}
    chkShowMenu.Checked := FManager.ShowMenu;
    chkSameType.Checked := not FManager.SameType;
    mmoIgnoreProperties.Lines.Assign(FManager.IgnoreProperties);

    if FManager.GridFont <> nil then
      pnlFont.Font := FManager.GridFont;
{$ENDIF}

    if ShowModal = mrOK then
    begin
{$IFNDEF STAND_ALONE}
      if FontChanged then
      begin
        if FManager.GridFont = nil then
          FManager.GridFont := TFont.Create;
        FManager.GridFont := pnlFont.Font;
      end;
      FManager.ShowMenu := chkShowMenu.Checked;
      FManager.SameType := not chkSameType.Checked;
      FManager.IgnoreProperties.Assign(mmoIgnoreProperties.Lines);

      UpdateFont;
{$ENDIF}
    end;

    Free;
  end;
end;

procedure TCnPropertyCompareForm.actListLeftExecute(Sender: TObject);
{$IFNDEF STAND_ALONE}
var
  R: TObject;
  Ini: TCustomIniFile;
{$ENDIF}
begin
{$IFNDEF STAND_ALONE}
  Ini := CreateWizardIni;
  try
    R := CnListComponentForOne(Ini);
    if R <> nil then
    begin
      LeftObject := R;
      LoadListProperties;
      ShowProperties;
    end;
  finally
    Ini.Free;
  end;
{$ENDIF}
end;

procedure TCnPropertyCompareForm.actListRightExecute(Sender: TObject);
{$IFNDEF STAND_ALONE}
var
  R: TObject;
  Ini: TCustomIniFile;
{$ENDIF}
begin
{$IFNDEF STAND_ALONE}
  Ini := CreateWizardIni;
  try
    R := CnListComponentForOne(Ini);
    if R <> nil then
    begin
      RightObject := R;
      LoadListProperties;
      ShowProperties;
    end;
  finally
    Ini.Free;
  end;
{$ENDIF}
end;

{$IFNDEF STAND_ALONE}

function TCnPropertyCompareForm.CreateWizardIni: TCustomIniFile;
var
  Wizard: TCnBaseWizard;
begin
  Result := nil;
  Wizard := CnWizardMgr.WizardByClassName('TCnDesignWizard');
  if Wizard <> nil then
    Result := Wizard.CreateIniFile;
end;

{$ENDIF}

procedure TCnPropertyCompareForm.actOnlyDiffExecute(Sender: TObject);
begin
  FOnlyDiff := not FOnlyDiff;
  actOnlyDiff.Checked := FOnlyDiff;

{$IFNDEF STAND_ALONE}
  if FManager <> nil then
    FManager.OnlyShowDiff := FOnlyDiff;
{$ENDIF}

  LoadListProperties;
  ShowProperties;
end;

procedure TCnPropertyCompareForm.actShowEventsExecute(Sender: TObject);
begin
  FShowEvents := not FShowEvents;
  actShowEvents.Checked := FShowEvents;

{$IFNDEF STAND_ALONE}
  if FManager <> nil then
    FManager.ShowEvents := FShowEvents;
{$ENDIF}

  LoadListProperties;
  ShowProperties;
end;

procedure TCnPropertyCompareForm.UpdateFont;
{$IFNDEF STAND_ALONE}
var
  H: Integer;
{$ENDIF}
begin
{$IFNDEF STAND_ALONE}
  if FManager.GridFont = nil then
    Exit;

  gridLeft.Font := FManager.GridFont;
  gridRight.Font := FManager.GridFont;

{$IFDEF DEBUG}
  CnDebugger.LogFmt('PropertyCompare Grid Font: %s Size %d. Height %d.', [FManager.GridFont.Name,
    FManager.GridFont.Size, FManager.GridFont.Height]);
{$ENDIF}

  H := FManager.GridFont.Height + 8;
  if H < 18 then
    H := 18;

  gridLeft.DefaultRowHeight := H;
  gridRight.DefaultRowHeight := H;
{$ENDIF}
end;

{$ENDIF CNWIZARDS_CNDESIGNWIZARD}
end.
