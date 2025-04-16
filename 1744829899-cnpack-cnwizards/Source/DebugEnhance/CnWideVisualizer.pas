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

unit CnWideVisualizer;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ���� UnicodeString/WideString ���ڴ�������ص���ĵ����ڲ鿴��
* ��Ԫ���ߣ�CnPack ������
* ��    ע���ṹ�ο��� VCL ���Դ��ĸ��� Visualizer
* ����ƽ̨��PWin11 + Delphi 12
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2024.04.04 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  SysUtils, Classes, Graphics, Controls, Forms, Messages, Dialogs, ComCtrls,
  StdCtrls, Grids, ExtCtrls, ToolsAPI, CnWizConsts, CnWizDebuggerNotifier,
  CnWizUtils, CnWizMultiLang, CnWizMultiLangFrame, CnWizIdeDock, CnHexEditor;

type
  TCnWideViewerFrame = class(TCnTranslateFrame {$IFDEF IDE_HAS_DEBUGGERVISUALIZER},
    IOTADebuggerVisualizerExternalViewerUpdater {$ENDIF})
    Panel1: TPanel;
    pgcView: TPageControl;
    tsHex: TTabSheet;
    tsWide: TTabSheet;
    mmoWide: TMemo;
  private
    FHexEditor: TCnHexEditor;
    FExpression: string;
    FOwningForm: TCustomForm;
{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}
    FClosedProc: TOTAVisualizerClosedProcedure;
{$ENDIF}
    FItems: TStrings;
    FAvailableState: TCnAvailableState;
    FEvaluator: TCnRemoteProcessEvaluator;
    procedure SetForm(AForm: TCustomForm);
    procedure AddWideContent(const Expression, TypeName, EvalResult: string; IsCpp: Boolean = False);
    procedure SetAvailableState(const AState: TCnAvailableState);
    procedure Clear;
{$IFDEF DELPHI120_ATHENS_UP}
    procedure WMDPIChangedAfterParent(var Message: TMessage); message WM_DPICHANGED_AFTERPARENT;
{$ENDIF}
  protected
    procedure SetParent(AParent: TWinControl); override;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}
    { IOTADebuggerVisualizerExternalViewerUpdater }
    procedure CloseVisualizer;
    procedure MarkUnavailable(Reason: TOTAVisualizerUnavailableReason);
    procedure RefreshVisualizer(const Expression, TypeName, EvalResult: string);
    procedure SetClosedCallback(ClosedProc: TOTAVisualizerClosedProcedure);
{$ENDIF}
  end;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

  TCnDebuggerWideVisualizer = class(TInterfacedObject, IOTADebuggerVisualizer,
    {$IFDEF FULL_IOTADEBUGGERVISUALIZER_250} IOTADebuggerVisualizer250, {$ENDIF}
    IOTADebuggerVisualizerExternalViewer)
  public
    { IOTADebuggerVisualizer }
    function GetSupportedTypeCount: Integer;
    procedure GetSupportedType(Index: Integer; var TypeName: string;
      var AllDescendants: Boolean); {$IFDEF FULL_IOTADEBUGGERVISUALIZER_250} overload; {$ENDIF}
    function GetVisualizerIdentifier: string;
    function GetVisualizerName: string;
    function GetVisualizerDescription: string;
{$IFDEF FULL_IOTADEBUGGERVISUALIZER_250}
    { IOTADebuggerVisualizer250 }
    procedure GetSupportedType(Index: Integer; var TypeName: string;
      var AllDescendants: Boolean; var IsGeneric: Boolean); overload;
{$ENDIF}
    { IOTADebuggerVisualizerExternalViewer }
    function GetMenuText: string;
    function Show(const Expression, TypeName, EvalResult: string;
      SuggestedLeft, SuggestedTop: Integer): IOTADebuggerVisualizerExternalViewerUpdater;
  end;

{$ENDIF}

procedure ShowWideExternalViewer(const Expression: string);
{* ���ֹ����õķ�ʽ����һ�������� TBytes �ı��ʽ����ʾ������ Delphi �������ʾ��ť}

implementation

uses
  {$IFDEF COMPILER6_UP} DesignIntf, {$ELSE} DsgnIntf, {$ENDIF}
   Actnlist, ImgList, Menus, IniFiles, CnCommon,
  {$IFDEF IDE_SUPPORT_THEMING} GraphUtil, {$ENDIF}
  {$IFDEF DELPHI103_RIO_UP} BrandingAPI, {$ENDIF}
  CnLangMgr, CnWizIdeUtils, CnNative {$IFDEF DEBUG}, CnDebug {$ENDIF};

{$R *.DFM}

const
  MAX_LEN = $10000;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

type
  ICnFrameFormHelper = interface
    ['{0FD4A98F-CE6B-422A-BF13-14E59707D3B2}']
    function GetForm: TCustomForm;
    function GetFrame: TCustomFrame;
    procedure SetForm(Form: TCustomForm);
    procedure SetFrame(Form: TCustomFrame);
  end;

  TCnWideVisualizerForm = class(TInterfacedObject, INTACustomDockableForm, ICnFrameFormHelper)
  private
    FMyFrame: TCnWideViewerFrame;
    FMyForm: TCustomForm;
    FExpression: string;
  public
    constructor Create(const Expression: string);
    { INTACustomDockableForm }
    function GetCaption: string;
    function GetFrameClass: TCustomFrameClass;
    procedure FrameCreated(AFrame: TCustomFrame);
    function GetIdentifier: string;
    function GetMenuActionList: TCustomActionList;
    function GetMenuImageList: TCustomImageList;
    procedure CustomizePopupMenu(PopupMenu: TPopupMenu);
    function GetToolbarActionList: TCustomActionList;
    function GetToolbarImageList: TCustomImageList;
    procedure CustomizeToolBar(ToolBar: TToolBar);
    procedure LoadWindowState(Desktop: TCustomIniFile; const Section: string);
    procedure SaveWindowState(Desktop: TCustomIniFile; const Section: string; IsProject: Boolean);
    function GetEditState: TEditState;
    function EditAction(Action: TEditAction): Boolean;
    { IFrameFormHelper }
    function GetForm: TCustomForm;
    function GetFrame: TCustomFrame;
    procedure SetForm(Form: TCustomForm);
    procedure SetFrame(Frame: TCustomFrame);
  end;

{$ENDIF}

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

{ TCnDebuggerWideVisualizer }

function TCnDebuggerWideVisualizer.GetMenuText: string;
begin
  Result := SCnDebugWideViewerMenuText;
end;

function TCnDebuggerWideVisualizer.GetSupportedTypeCount: Integer;
begin
  Result := 7;
end;

procedure TCnDebuggerWideVisualizer.GetSupportedType(Index: Integer; var TypeName: string;
  var AllDescendants: Boolean);
begin
  AllDescendants := False;
  case Index of
    0: TypeName := 'string';
    1: TypeName := 'UnicodeString';
    2: TypeName := 'WideString';
    3: TypeName := 'array of Char';
    4: TypeName := 'array of WideChar';
    5: TypeName := 'TArray<Char>';
    6: TypeName := 'TArray<WideChar>';
  end;
{$IFDEF DEBUG}
  CnDebugger.LogFmt('WideVisualizer.GetSupportedType #%d: %s', [Index, TypeName])
{$ENDIF}
end;

{$IFDEF FULL_IOTADEBUGGERVISUALIZER_250}

procedure TCnDebuggerWideVisualizer.GetSupportedType(Index: Integer;
  var TypeName: string; var AllDescendants, IsGeneric: Boolean);
begin
  GetSupportedType(Index, TypeName, AllDescendants);
{$IFDEF IDE_HAS_MEMORY_VISUALIZAER}
  IsGeneric := False; // �� IDE ��ϵͳ�� Memory Viewer ������£��˴����� True �������� TArray<TObject> Ҳ���ַŴ󾵴Ӷ�����
{$ELSE}
  IsGeneric := Index in [5, 6];
{$ENDIF}
end;

{$ENDIF}

function TCnDebuggerWideVisualizer.GetVisualizerDescription: string;
begin
  Result := SCnDebugWideViewerDescription;
end;

function TCnDebuggerWideVisualizer.GetVisualizerIdentifier: string;
begin
  Result := ClassName;
end;

function TCnDebuggerWideVisualizer.GetVisualizerName: string;
begin
  Result := SCnDebugWideViewerName;
end;

function TCnDebuggerWideVisualizer.Show(const Expression, TypeName, EvalResult: string;
  SuggestedLeft, SuggestedTop: Integer): IOTADebuggerVisualizerExternalViewerUpdater;
var
  AForm: TCustomForm;
  AFrame: TCnWideViewerFrame;
  VisDockForm: INTACustomDockableForm;
{$IFDEF IDE_SUPPORT_THEMING}
  LThemingServices: IOTAIDEThemingServices;
{$ENDIF}
begin
  CloseExpandableEvalViewForm; // ������ʾ���ڿ��ܹ���ס�����ڣ�������֮����Ҳ��

  VisDockForm := TCnWideVisualizerForm.Create(Expression) as INTACustomDockableForm;
  AForm := (BorlandIDEServices as INTAServices).CreateDockableForm(VisDockForm);

{$IFDEF DELPHI120_ATHENS_UP}
  AForm.LockDrawing;
  try
{$ENDIF}
    AForm.Left := SuggestedLeft;
    AForm.Top := SuggestedTop;
    (VisDockForm as ICnFrameFormHelper).SetForm(AForm);
    AFrame := (VisDockForm as ICnFrameFormHelper).GetFrame as TCnWideViewerFrame;
    AFrame.AddWideContent(Expression, TypeName, EvalResult, CurrentIsCSource);

    Result := AFrame as IOTADebuggerVisualizerExternalViewerUpdater;
{$IFDEF IDE_SUPPORT_THEMING}
    if Supports(BorlandIDEServices, IOTAIDEThemingServices, LThemingServices) and
      LThemingServices.IDEThemingEnabled then
    begin
      AFrame.Panel1.StyleElements := AFrame.Panel1.StyleElements - [seClient];
      AFrame.Panel1.ParentBackground := False;
      LThemingServices.ApplyTheme(AForm);
  {$IFDEF DELPHI103_RIO_UP}
      AFrame.Panel1.Color := ColorBlendRGB(LThemingServices.StyleServices.GetSystemColor(clWindowText),
      LThemingServices.StyleServices.GetSystemColor(clWindow), 0.5);
  {$ENDIF}
{$IFDEF DELPHI120_ATHENS_UP}
      if TIDEThemeMetrics.Font.Enabled then
        AFrame.Font.Assign(TIDEThemeMetrics.Font.GetFont());
{$ENDIF}
    end;
{$ENDIF}
{$IFDEF DELPHI120_ATHENS_UP}
  finally
    AForm.UnlockDrawing;
  end;
{$ENDIF}
end;

{$ENDIF}

{ TCnWideViewerFrame }

procedure TCnWideViewerFrame.SetAvailableState(const AState: TCnAvailableState);
var
  S: string;
begin
  FAvailableState := AState;
  case FAvailableState of
    asAvailable:
      ;
    asProcRunning:
      S := SCnDebugErrorProcessNotAccessible;
    asOutOfScope:
      S := SCnDebugErrorOutOfScope;
    asNotAvailable:
      S := SCnDebugErrorValueNotAccessible;
  end;

  if S <> '' then
  begin
    Clear;
    Caption := S;
  end;
end;

procedure TCnWideViewerFrame.AddWideContent(const Expression, TypeName,
  EvalResult: string; IsCpp: Boolean);
var
  DebugSvcs: IOTADebuggerServices;
  CurProcess: IOTAProcess;
  CurThread: IOTAThread;
  S, PE, LE: string; // ��������͡�ָ����ʽ�����ȱ��ʽ
  P, L: TUInt64;
  Buf: TBytes;
  M: WideString;
begin
  if Supports(BorlandIDEServices, IOTADebuggerServices, DebugSvcs) then
    CurProcess := DebugSvcs.CurrentProcess;
  if CurProcess = nil then
    Exit;
  CurThread := CurProcess.CurrentThread;
  if CurThread = nil then
    Exit;

{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnWideViewerFrame.AddWideContent: %s: %s', [Expression, TypeName]);
{$ENDIF}

  FExpression := Expression;
  SetAvailableState(asAvailable);

  Clear;
  PE := Format('Pointer(%s)', [Expression]);
  LE := Format('Length(%s)', [Expression]);

  S := FEvaluator.EvaluateExpression(LE);
  L := StrToIntDef(S, 0);
  if L <= 0 then // �������Ϊ 0 ��ɶ������ʾ���˳�
    Exit;
  if L > MAX_LEN then // ����̫��
    L := MAX_LEN;

  S := FEvaluator.EvaluateExpression(PE);
  if S = '' then // ����˵��û���õ�ָ���ַ������ֱ����ֵ
  begin
    // ����ǳ�����Pointer �ò�����ֱ��ȡֵ
    S := FEvaluator.EvaluateExpression(Expression);
    if S = '' then
      Exit;

{$IFDEF UNICODE}
    SetLength(Buf, Length(S) * SizeOf(WideChar));
    Move(S[1], Buf[0], Length(Buf));
    mmoWide.Lines.Text := S;
{$ELSE}
    M := WideString(S);
    SetLength(Buf, Length(M) * SizeOf(WideChar));
    Move(M[1], Buf[0], Length(Buf));
    mmoWide.Lines.Text := M;
{$ENDIF}
    FHexEditor.LoadFromBuffer(Buf[0], Length(Buf));
  end
  else
  begin
    P := StrToUInt64(S);
    SetLength(Buf, L * SizeOf(WideChar));
    CurProcess.ReadProcessMemory(P, L * SizeOf(WideChar), Buf[0]);
    FHexEditor.LoadFromBuffer(Buf[0], Length(Buf));

    SetLength(M, L);
    Move(Buf[0], M[1], Length(Buf));
    mmoWide.Lines.Text := M;
  end;
end;

procedure TCnWideViewerFrame.Clear;
begin
  FHexEditor.Clear;
  mmoWide.Lines.Clear;
end;

constructor TCnWideViewerFrame.Create(AOwner: TComponent);
begin
  inherited;
  FEvaluator := TCnRemoteProcessEvaluator.Create;
  FHexEditor := TCnHexEditor.Create(Self);
  FHexEditor.Align := alClient;
  FHexEditor.Parent := tsHex;
end;

destructor TCnWideViewerFrame.Destroy;
begin
  FEvaluator.Free;
  inherited;
end;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

procedure TCnWideViewerFrame.CloseVisualizer;
begin
  if FOwningForm <> nil then
    FOwningForm.Close;
end;

procedure TCnWideViewerFrame.MarkUnavailable(
  Reason: TOTAVisualizerUnavailableReason);
begin
  if Reason = ovurProcessRunning then
    SetAvailableState(asProcRunning)
  else if Reason = ovurOutOfScope then
    SetAvailableState(asOutOfScope);
end;

procedure TCnWideViewerFrame.RefreshVisualizer(const Expression, TypeName,
  EvalResult: string);
begin
  AddWideContent(Expression, TypeName, EvalResult, CurrentIsCSource);
end;

procedure TCnWideViewerFrame.SetClosedCallback(
  ClosedProc: TOTAVisualizerClosedProcedure);
begin
  FClosedProc := ClosedProc;
end;

{$ENDIF}

procedure TCnWideViewerFrame.SetForm(AForm: TCustomForm);
begin
  FOwningForm := AForm;
end;

procedure TCnWideViewerFrame.SetParent(AParent: TWinControl);
begin
  if AParent = nil then
  begin
    FreeAndNil(FItems);
{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}
    if Assigned(FClosedProc) then
      FClosedProc;
{$ENDIF}
  end;
  inherited;
end;

{$IFDEF DELPHI120_ATHENS_UP}

procedure TCnWideViewerFrame.WMDPIChangedAfterParent(var Message: TMessage);
begin
  inherited;
  if TIDEThemeMetrics.Font.Enabled then
    TIDEThemeMetrics.Font.AdjustDPISize(Font, TIDEThemeMetrics.Font.Size, PixelsPerInch);
end;

{$ENDIF}

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

{ TCnWideVisualizerForm }

constructor TCnWideVisualizerForm.Create(const Expression: string);
begin
  inherited Create;
  FExpression := Expression;
end;

procedure TCnWideVisualizerForm.CustomizePopupMenu(PopupMenu: TPopupMenu);
begin
  // no toolbar
end;

procedure TCnWideVisualizerForm.CustomizeToolBar(ToolBar: TToolBar);
begin
 // no toolbar
end;

function TCnWideVisualizerForm.EditAction(Action: TEditAction): Boolean;
begin
  Result := False;
end;

procedure TCnWideVisualizerForm.FrameCreated(AFrame: TCustomFrame);
begin
  FMyFrame := TCnWideViewerFrame(AFrame);
end;

function TCnWideVisualizerForm.GetCaption: string;
begin
  Result := Format(SCnWideViewerFormCaption, [FExpression]);
end;

function TCnWideVisualizerForm.GetEditState: TEditState;
begin
  Result := [];
end;

function TCnWideVisualizerForm.GetForm: TCustomForm;
begin
  Result := FMyForm;
end;

function TCnWideVisualizerForm.GetFrame: TCustomFrame;
begin
  Result := FMyFrame;
end;

function TCnWideVisualizerForm.GetFrameClass: TCustomFrameClass;
begin
  Result := TCnWideViewerFrame;
end;

function TCnWideVisualizerForm.GetIdentifier: string;
begin
  Result := 'WideDebugVisualizer';
end;

function TCnWideVisualizerForm.GetMenuActionList: TCustomActionList;
begin
  Result := nil;
end;

function TCnWideVisualizerForm.GetMenuImageList: TCustomImageList;
begin
  Result := nil;
end;

function TCnWideVisualizerForm.GetToolbarActionList: TCustomActionList;
begin
  Result := nil;
end;

function TCnWideVisualizerForm.GetToolbarImageList: TCustomImageList;
begin
  Result := nil;
end;

procedure TCnWideVisualizerForm.LoadWindowState(Desktop: TCustomIniFile;
  const Section: string);
begin
  // no desktop saving
end;

procedure TCnWideVisualizerForm.SaveWindowState(Desktop: TCustomIniFile;
  const Section: string; IsProject: Boolean);
begin
  // no desktop saving
end;

procedure TCnWideVisualizerForm.SetForm(Form: TCustomForm);
begin
  FMyForm := Form;
  if Assigned(FMyFrame) then
    FMyFrame.SetForm(FMyForm);
end;

procedure TCnWideVisualizerForm.SetFrame(Frame: TCustomFrame);
begin
   FMyFrame := TCnWideViewerFrame(Frame);
end;

{$ENDIF}

procedure ShowWideExternalViewer(const Expression: string);
var
  F: TCnIdeDockForm;
  Fm: TCnWideViewerFrame;
  S: string;
begin
  // ��ֵ��Ҫ�����õ� Length() ֵ
  S := CnRemoteProcessEvaluator.EvaluateExpression(Format('Length(%s)', [Expression]));
  if S = '' then
  begin
    ErrorDlg(Format(SCnDebugErrorExprNotAClass, [Expression, 'WideString/UnicodeString']));
    Exit;
  end;

  F := TCnIdeDockForm.Create(Application);
  F.Caption := Format(SCnWideViewerFormCaption, [Expression]);
  Fm := TCnWideViewerFrame.Create(F);
  Fm.SetForm(F);
  Fm.Parent := F;
  Fm.Align := alClient;
  Fm.AddWideContent(Expression, '', '',  CurrentIsCSource); // ���� C/C++ ���� Pascal Ϊ׼

  F.Show;
end;

end.

