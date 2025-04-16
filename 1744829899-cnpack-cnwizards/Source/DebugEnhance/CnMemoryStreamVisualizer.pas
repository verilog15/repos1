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

unit CnMemoryStreamVisualizer;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ���� MemoryStream ���ڴ�������ص���ĵ����ڲ鿴��
* ��Ԫ���ߣ�CnPack ������
* ��    ע���ṹ�ο��� VCL ���Դ��ĸ��� Visualizer
* ����ƽ̨��PWin11 + Delphi 12
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2024.04.05 V1.0
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
  TCnMemoryStreamViewerFrame = class(TCnTranslateFrame {$IFDEF IDE_HAS_DEBUGGERVISUALIZER},
    IOTADebuggerVisualizerExternalViewerUpdater {$ENDIF})
    Panel1: TPanel;
    pgcView: TPageControl;
    tsHex: TTabSheet;
    tsAnsi: TTabSheet;
    tsWide: TTabSheet;
    mmoAnsi: TMemo;
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
    procedure AddMemoryStreamContent(const Expression, TypeName, EvalResult: string; IsCpp: Boolean = False);
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

  TCnDebuggerMemoryStreamVisualizer = class(TInterfacedObject, IOTADebuggerVisualizer,
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

procedure ShowMemoryStreamExternalViewer(const Expression: string);
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
  MAX_LEN = $1000;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

type
  ICnFrameFormHelper = interface
    ['{0FD4A98F-CE6B-422A-BF13-14E59707D3B2}']
    function GetForm: TCustomForm;
    function GetFrame: TCustomFrame;
    procedure SetForm(Form: TCustomForm);
    procedure SetFrame(Form: TCustomFrame);
  end;

  TCnMemoryStreamVisualizerForm = class(TInterfacedObject, INTACustomDockableForm, ICnFrameFormHelper)
  private
    FMyFrame: TCnMemoryStreamViewerFrame;
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

{ TCnDebuggerMemoryStreamVisualizer }

function TCnDebuggerMemoryStreamVisualizer.GetMenuText: string;
begin
  Result := SCnDebugMemoryStreamViewerMenuText;
end;

function TCnDebuggerMemoryStreamVisualizer.GetSupportedTypeCount: Integer;
begin
  Result := 1;
end;

procedure TCnDebuggerMemoryStreamVisualizer.GetSupportedType(Index: Integer; var TypeName: string;
  var AllDescendants: Boolean);
begin
  AllDescendants := True;
  case Index of
    0: TypeName := 'TCustomMemoryStream';
  end;
{$IFDEF DEBUG}
  CnDebugger.LogFmt('MemoryStreamVisualizer.GetSupportedType #%d: %s', [Index, TypeName])
{$ENDIF}
end;

{$IFDEF FULL_IOTADEBUGGERVISUALIZER_250}

procedure TCnDebuggerMemoryStreamVisualizer.GetSupportedType(Index: Integer;
  var TypeName: string; var AllDescendants, IsGeneric: Boolean);
begin
  GetSupportedType(Index, TypeName, AllDescendants);
  IsGeneric := False;
end;

{$ENDIF}

function TCnDebuggerMemoryStreamVisualizer.GetVisualizerDescription: string;
begin
  Result := SCnDebugMemoryStreamViewerDescription;
end;

function TCnDebuggerMemoryStreamVisualizer.GetVisualizerIdentifier: string;
begin
  Result := ClassName;
end;

function TCnDebuggerMemoryStreamVisualizer.GetVisualizerName: string;
begin
  Result := SCnDebugMemoryStreamViewerName;
end;

function TCnDebuggerMemoryStreamVisualizer.Show(const Expression, TypeName, EvalResult: string;
  SuggestedLeft, SuggestedTop: Integer): IOTADebuggerVisualizerExternalViewerUpdater;
var
  AForm: TCustomForm;
  AFrame: TCnMemoryStreamViewerFrame;
  VisDockForm: INTACustomDockableForm;
{$IFDEF IDE_SUPPORT_THEMING}
  LThemingServices: IOTAIDEThemingServices;
{$ENDIF}
begin
  CloseExpandableEvalViewForm; // ������ʾ���ڿ��ܹ���ס�����ڣ�������֮����Ҳ��

  VisDockForm := TCnMemoryStreamVisualizerForm.Create(Expression) as INTACustomDockableForm;
  AForm := (BorlandIDEServices as INTAServices).CreateDockableForm(VisDockForm);

{$IFDEF DELPHI120_ATHENS_UP}
  AForm.LockDrawing;
  try
{$ENDIF}
    AForm.Left := SuggestedLeft;
    AForm.Top := SuggestedTop;
    (VisDockForm as ICnFrameFormHelper).SetForm(AForm);
    AFrame := (VisDockForm as ICnFrameFormHelper).GetFrame as TCnMemoryStreamViewerFrame;
    AFrame.AddMemoryStreamContent(Expression, TypeName, EvalResult, CurrentIsCSource);

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

{ TCnMemoryStreamViewerFrame }

procedure TCnMemoryStreamViewerFrame.SetAvailableState(const AState: TCnAvailableState);
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

procedure TCnMemoryStreamViewerFrame.AddMemoryStreamContent(const Expression, TypeName,
  EvalResult: string; IsCpp: Boolean);
var
  DebugSvcs: IOTADebuggerServices;
  CurProcess: IOTAProcess;
  CurThread: IOTAThread;
  S, PE, LE: string; // ��������͡�ָ����ʽ�����ȱ��ʽ
  P, L: TUInt64;
  Buf: TBytes;
  M: AnsiString;
  W: WideString;
begin
  if Supports(BorlandIDEServices, IOTADebuggerServices, DebugSvcs) then
    CurProcess := DebugSvcs.CurrentProcess;
  if CurProcess = nil then
    Exit;
  CurThread := CurProcess.CurrentThread;
  if CurThread = nil then
    Exit;

{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnMemoryStreamViewerFrame.AddMemoryStreamContent: %s: %s', [Expression, TypeName]);
{$ENDIF}

  FExpression := Expression;
  SetAvailableState(asAvailable);

  Clear;
  PE := Format('(TCustomMemoryStream(%s)).Memory', [Expression]);
  LE := Format('(TCustomMemoryStream(%s)).Size', [Expression]);

  S := FEvaluator.EvaluateExpression(LE);
  L := StrToIntDef(S, 0);
  if L <= 0 then // �������Ϊ 0 ��ɶ������ʾ���˳�
    Exit;
  if L > MAX_LEN then // ����̫��
    L := MAX_LEN;

  S := FEvaluator.EvaluateExpression(PE);
  if (S = '') or (S = 'nil') then // �����յ�ָ�룬Ҳû����ʾ
    Exit;

  P := StrToUInt64(S);
  SetLength(Buf, L);
  CurProcess.ReadProcessMemory(P, L, Buf[0]);
  FHexEditor.LoadFromBuffer(Buf[0], Length(Buf));

  SetLength(M, L);
  Move(Buf[0], M[1], Length(Buf));
  mmoAnsi.Lines.Text := M;

  L := L shr 1;
  SetLength(W, L);
  Move(Buf[0], W[1], L * SizeOf(WideChar));
  mmoWide.Lines.Text := W;
end;

procedure TCnMemoryStreamViewerFrame.Clear;
begin
  FHexEditor.Clear;
  mmoAnsi.Lines.Clear;
  mmoWide.Lines.Clear;
end;

constructor TCnMemoryStreamViewerFrame.Create(AOwner: TComponent);
begin
  inherited;
  FEvaluator := TCnRemoteProcessEvaluator.Create;
  FHexEditor := TCnHexEditor.Create(Self);
  FHexEditor.Align := alClient;
  FHexEditor.Parent := tsHex;
end;

destructor TCnMemoryStreamViewerFrame.Destroy;
begin
  FEvaluator.Free;
  inherited;
end;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

procedure TCnMemoryStreamViewerFrame.CloseVisualizer;
begin
  if FOwningForm <> nil then
    FOwningForm.Close;
end;

procedure TCnMemoryStreamViewerFrame.MarkUnavailable(
  Reason: TOTAVisualizerUnavailableReason);
begin
  if Reason = ovurProcessRunning then
    SetAvailableState(asProcRunning)
  else if Reason = ovurOutOfScope then
    SetAvailableState(asOutOfScope);
end;

procedure TCnMemoryStreamViewerFrame.RefreshVisualizer(const Expression, TypeName,
  EvalResult: string);
begin
  AddMemoryStreamContent(Expression, TypeName, EvalResult, CurrentIsCSource);
end;

procedure TCnMemoryStreamViewerFrame.SetClosedCallback(
  ClosedProc: TOTAVisualizerClosedProcedure);
begin
  FClosedProc := ClosedProc;
end;

{$ENDIF}

procedure TCnMemoryStreamViewerFrame.SetForm(AForm: TCustomForm);
begin
  FOwningForm := AForm;
end;

procedure TCnMemoryStreamViewerFrame.SetParent(AParent: TWinControl);
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

procedure TCnMemoryStreamViewerFrame.WMDPIChangedAfterParent(var Message: TMessage);
begin
  inherited;
  if TIDEThemeMetrics.Font.Enabled then
    TIDEThemeMetrics.Font.AdjustDPISize(Font, TIDEThemeMetrics.Font.Size, PixelsPerInch);
end;

{$ENDIF}

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

{ TCnMemoryStreamVisualizerForm }

constructor TCnMemoryStreamVisualizerForm.Create(const Expression: string);
begin
  inherited Create;
  FExpression := Expression;
end;

procedure TCnMemoryStreamVisualizerForm.CustomizePopupMenu(PopupMenu: TPopupMenu);
begin
  // no toolbar
end;

procedure TCnMemoryStreamVisualizerForm.CustomizeToolBar(ToolBar: TToolBar);
begin
 // no toolbar
end;

function TCnMemoryStreamVisualizerForm.EditAction(Action: TEditAction): Boolean;
begin
  Result := False;
end;

procedure TCnMemoryStreamVisualizerForm.FrameCreated(AFrame: TCustomFrame);
begin
  FMyFrame := TCnMemoryStreamViewerFrame(AFrame);
end;

function TCnMemoryStreamVisualizerForm.GetCaption: string;
begin
  Result := Format(SCnMemoryStreamViewerFormCaption, [FExpression]);
end;

function TCnMemoryStreamVisualizerForm.GetEditState: TEditState;
begin
  Result := [];
end;

function TCnMemoryStreamVisualizerForm.GetForm: TCustomForm;
begin
  Result := FMyForm;
end;

function TCnMemoryStreamVisualizerForm.GetFrame: TCustomFrame;
begin
  Result := FMyFrame;
end;

function TCnMemoryStreamVisualizerForm.GetFrameClass: TCustomFrameClass;
begin
  Result := TCnMemoryStreamViewerFrame;
end;

function TCnMemoryStreamVisualizerForm.GetIdentifier: string;
begin
  Result := 'MemoryStreamDebugVisualizer';
end;

function TCnMemoryStreamVisualizerForm.GetMenuActionList: TCustomActionList;
begin
  Result := nil;
end;

function TCnMemoryStreamVisualizerForm.GetMenuImageList: TCustomImageList;
begin
  Result := nil;
end;

function TCnMemoryStreamVisualizerForm.GetToolbarActionList: TCustomActionList;
begin
  Result := nil;
end;

function TCnMemoryStreamVisualizerForm.GetToolbarImageList: TCustomImageList;
begin
  Result := nil;
end;

procedure TCnMemoryStreamVisualizerForm.LoadWindowState(Desktop: TCustomIniFile;
  const Section: string);
begin
  // no desktop saving
end;

procedure TCnMemoryStreamVisualizerForm.SaveWindowState(Desktop: TCustomIniFile;
  const Section: string; IsProject: Boolean);
begin
  // no desktop saving
end;

procedure TCnMemoryStreamVisualizerForm.SetForm(Form: TCustomForm);
begin
  FMyForm := Form;
  if Assigned(FMyFrame) then
    FMyFrame.SetForm(FMyForm);
end;

procedure TCnMemoryStreamVisualizerForm.SetFrame(Frame: TCustomFrame);
begin
   FMyFrame := TCnMemoryStreamViewerFrame(Frame);
end;

{$ENDIF}

procedure ShowMemoryStreamExternalViewer(const Expression: string);
var
  F: TCnIdeDockForm;
  Fm: TCnMemoryStreamViewerFrame;
begin
  if not CnWizDebuggerObjectInheritsFrom(Expression, 'TCustomMemoryStream') then
  begin
    ErrorDlg(Format(SCnDebugErrorExprNotAClass, [Expression, 'TCustomMemoryStream']));
    Exit;
  end;

  F := TCnIdeDockForm.Create(Application);
  F.Caption := Format(SCnMemoryStreamViewerFormCaption, [Expression]);
  Fm := TCnMemoryStreamViewerFrame.Create(F);
  Fm.SetForm(F);
  Fm.Parent := F;
  Fm.Align := alClient;
  Fm.AddMemoryStreamContent(Expression, '', '',  CurrentIsCSource); // ���� C/C++ ���� Pascal Ϊ׼

  F.Show;
end;

end.

