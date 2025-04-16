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

unit CnDataSetVisualizer;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ���� TDataSet ��������ĵ����ڲ鿴��
* ��Ԫ���ߣ�CnPack������
* ��    ע���ṹ�ο��� VCL ���Դ��ĸ��� Visualizer
* ����ƽ̨��PWin11 + Delphi 12
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2024.03.30 V1.0
*               IOTADebuggerVisualizer250 �ĳ� 10.3 ��֧�֣����� 10.2 �� Update ����֧��
*           2024.03.09 V1.1
*               �ع��Գ�ȡ��ֵ�����ⲿ
*           2024.03.07 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  SysUtils, Classes, Graphics, Controls, Forms, Messages, Dialogs, ComCtrls,
  StdCtrls, Grids, ExtCtrls, ToolsAPI, CnWizConsts, CnWizDebuggerNotifier,
  CnWizUtils, CnWizMultiLang, CnWizMultiLangFrame, CnWizIdeDock;

type
  TCnDataSetViewerFrame = class(TCnTranslateFrame {$IFDEF IDE_HAS_DEBUGGERVISUALIZER},
    IOTADebuggerVisualizerExternalViewerUpdater {$ENDIF})
  {* ��֧�ֵ��Կ��ӻ��ӿڵ� Delphi �£���ʵ������� IDE ������������
    ��֧�ֵĵͰ汾 Delphi �У�ͨ���˵�������д����������ڲ�Ƕ��� Frame ʵ��}
    pcViews: TPageControl;
    tsProp: TTabSheet;
    mmoProp: TMemo;
    tsData: TTabSheet;
    Panel1: TPanel;
    grdData: TStringGrid;
    tsField: TTabSheet;
    grdField: TStringGrid;
    procedure pcViewsChange(Sender: TObject);
  private
    FExpression: string;
    FOwningForm: TCustomForm;
{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}
    FClosedProc: TOTAVisualizerClosedProcedure;
{$ENDIF}
    FAvailableState: TCnAvailableState;
    FEvaluator: TCnRemoteProcessEvaluator;
    procedure SetForm(AForm: TCustomForm);
    procedure AddDataSetContent(const Expression, TypeName, EvalResult: string; IsCpp: Boolean = False);
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

  TCnDebuggerDataSetVisualizer = class(TInterfacedObject, IOTADebuggerVisualizer,
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

procedure ShowDataSetExternalViewer(const Expression: string);
{* ���ֹ����õķ�ʽ����һ�������� TDataSet �ı��ʽ����ʾ������ Delphi �������ʾ��ť}

implementation

uses
  {$IFDEF COMPILER6_UP} DesignIntf, {$ELSE} DsgnIntf, {$ENDIF}
   Actnlist, ImgList, Menus, IniFiles, CnCommon,
  {$IFDEF IDE_SUPPORT_THEMING} GraphUtil, {$ENDIF}
  {$IFDEF DELPHI103_RIO_UP} BrandingAPI, {$ENDIF}
  CnLangMgr, CnWizIdeUtils {$IFDEF DEBUG}, CnDebug {$ENDIF};

{$R *.DFM}

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

type
  ICnFrameFormHelper = interface
    ['{0FD4A98F-CE6B-422A-BF13-14E59707D3B2}']
    function GetForm: TCustomForm;
    function GetFrame: TCustomFrame;
    procedure SetForm(Form: TCustomForm);
    procedure SetFrame(Form: TCustomFrame);
  end;

  TCnDataSetVisualizerForm = class(TInterfacedObject, INTACustomDockableForm, ICnFrameFormHelper)
  private
    FMyFrame: TCnDataSetViewerFrame;
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

{ TCnDebuggerDataSetVisualizer }

function TCnDebuggerDataSetVisualizer.GetMenuText: string;
begin
  Result := SCnDebugDataSetViewerMenuText;
end;

function TCnDebuggerDataSetVisualizer.GetSupportedTypeCount: Integer;
begin
  Result := 1;
end;

procedure TCnDebuggerDataSetVisualizer.GetSupportedType(Index: Integer; var TypeName: string;
  var AllDescendants: Boolean);
begin
  TypeName := 'TDataSet';
  AllDescendants := True;
end;

{$IFDEF FULL_IOTADEBUGGERVISUALIZER_250}

procedure TCnDebuggerDataSetVisualizer.GetSupportedType(Index: Integer;
  var TypeName: string; var AllDescendants, IsGeneric: Boolean);
begin
  TypeName := 'TDataSet';
  AllDescendants := True;
  IsGeneric := False;
end;

{$ENDIF}

function TCnDebuggerDataSetVisualizer.GetVisualizerDescription: string;
begin
  Result := SCnDebugDataSetViewerDescription;
end;

function TCnDebuggerDataSetVisualizer.GetVisualizerIdentifier: string;
begin
  Result := ClassName;
end;

function TCnDebuggerDataSetVisualizer.GetVisualizerName: string;
begin
  Result := SCnDebugDataSetViewerName;
end;

function TCnDebuggerDataSetVisualizer.Show(const Expression, TypeName, EvalResult: string;
  SuggestedLeft, SuggestedTop: Integer): IOTADebuggerVisualizerExternalViewerUpdater;
var
  AForm: TCustomForm;
  AFrame: TCnDataSetViewerFrame;
  VisDockForm: INTACustomDockableForm;
{$IFDEF IDE_SUPPORT_THEMING}
  LThemingServices: IOTAIDEThemingServices;
{$ENDIF}
begin
  CloseExpandableEvalViewForm; // ������ʾ���ڿ��ܹ���ס�����ڣ�������֮����Ҳ��

  VisDockForm := TCnDataSetVisualizerForm.Create(Expression) as INTACustomDockableForm;
  AForm := (BorlandIDEServices as INTAServices).CreateDockableForm(VisDockForm);

{$IFDEF DELPHI120_ATHENS_UP}
  AForm.LockDrawing;
  try
{$ENDIF}
    AForm.Left := SuggestedLeft;
    AForm.Top := SuggestedTop;
    (VisDockForm as ICnFrameFormHelper).SetForm(AForm);
    AFrame := (VisDockForm as ICnFrameFormHelper).GetFrame as TCnDataSetViewerFrame;
    AFrame.AddDataSetContent(Expression, TypeName, EvalResult, CurrentIsCSource);
    AFrame.pcViewsChange(nil);
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

{ TCnDataSetViewerFrame }

procedure TCnDataSetViewerFrame.SetAvailableState(const AState: TCnAvailableState);
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
    mmoProp.Lines.Text := '';
end;

procedure TCnDataSetViewerFrame.AddDataSetContent(const Expression, TypeName,
  EvalResult: string; IsCpp: Boolean);
var
  DebugSvcs: IOTADebuggerServices;
  CurProcess: IOTAProcess;
  CurThread: IOTAThread;
  S: string;
  I, C: Integer;
begin
  if Supports(BorlandIDEServices, IOTADebuggerServices, DebugSvcs) then
    CurProcess := DebugSvcs.CurrentProcess;
  if CurProcess = nil then
    Exit;
  CurThread := CurProcess.CurrentThread;
  if CurThread = nil then
    Exit;

  FExpression := Expression;
  SetAvailableState(asAvailable);

  Clear;

  if IsCpp then
    S := FEvaluator.EvaluateExpression(FExpression + '->Active')
  else
    S := FEvaluator.EvaluateExpression(FExpression + '.Active');
  mmoProp.Lines.Add('Active: ' + S);

  if LowerCase(S) = 'true' then
  begin
    if IsCpp then
      S := FEvaluator.EvaluateExpression(FExpression + '->FieldCount')
    else
      S := FEvaluator.EvaluateExpression(FExpression + '.FieldCount');
    mmoProp.Lines.Add('FieldCount: ' + S);

    if IsCpp then
      S := FEvaluator.EvaluateExpression(FExpression + '->RecordCount')
    else
      S := FEvaluator.EvaluateExpression(FExpression + '.RecordCount');
    mmoProp.Lines.Add('RecordCount: ' + S);

    if IsCpp then
      S := FEvaluator.EvaluateExpression(FExpression + '->RecNo')
    else
      S := FEvaluator.EvaluateExpression(FExpression + '.RecNo');
    mmoProp.Lines.Add('RecNo: ' + S);

    // Fields Defs
    if IsCpp then
      S := FEvaluator.EvaluateExpression(FExpression+ '->FieldDefs->Count')
    else
      S := FEvaluator.EvaluateExpression(FExpression+ '.FieldDefs.Count');
    C := StrToIntDef(S, 0);

    grdField.RowCount := C + 1;
    grdField.FixedRows := 1;
    grdField.ColCount := 5;
    grdfield.FixedCols := 0;

    for I := 0 to grdField.ColCount - 1 do
      grdField.ColWidths[I] := 110;

    grdField.Cells[0, 0] := 'Name';
    grdField.Cells[1, 0] := 'DataType';
    grdField.Cells[2, 0] := 'Size';
    grdField.Cells[3, 0] := 'Precision';
    grdField.Cells[4, 0] := 'Attribute';

    for I := 0 to C - 1 do // ��ѭ��
    begin
      if IsCpp then
      begin
        grdField.Cells[0, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('->FieldDefs->Items[%d]->Name', [I]));
        grdField.Cells[1, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('->FieldDefs->Items[%d]->DataType', [I]));
        grdField.Cells[2, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('->FieldDefs->Items[%d]->Size', [I]));
        grdField.Cells[3, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('->FieldDefs->Items[%d]->Precision', [I]));
        grdField.Cells[4, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('->FieldDefs->Items[%d]->Attribute', [I]));
      end
      else
      begin
        grdField.Cells[0, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('.FieldDefs.Items[%d].Name', [I]));
        grdField.Cells[1, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('.FieldDefs.Items[%d].DataType', [I]));
        grdField.Cells[2, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('.FieldDefs.Items[%d].Size', [I]));
        grdField.Cells[3, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('.FieldDefs.Items[%d].Precision', [I]));
        grdField.Cells[4, I + 1] := FEvaluator.EvaluateExpression(FExpression + Format('.FieldDefs.Items[%d].Attribute', [I]));
      end;
    end;

    // Data
    grdData.ColCount := C;
    grdData.RowCount := 2;
    grdData.FixedRows := 1;
    for I := 0 to grdData.ColCount - 1 do
      grdData.ColWidths[I] := 90;

    for I := 0 to C - 1 do // ��ѭ������ӡ��ǰ��¼�ĸ��ֶ�ֵ
    begin
      if IsCpp then
      begin
        grdData.Cells[I, 0] := FEvaluator.EvaluateExpression(FExpression + Format('->FieldDefs->Items[%d]->Name', [I]));
        grdData.Cells[I, 1] := FEvaluator.EvaluateExpression(FExpression + Format('->Fields[%d]->AsString', [I]));
      end
      else
      begin
        grdData.Cells[I, 0] := FEvaluator.EvaluateExpression(FExpression + Format('.FieldDefs.Items[%d].Name', [I]));
        grdData.Cells[I, 1] := FEvaluator.EvaluateExpression(FExpression + Format('.Fields[%d].AsString', [I]));
      end;
    end;
  end;
end;

procedure TCnDataSetViewerFrame.Clear;
begin
  mmoProp.Lines.Clear;
  grdField.RowCount := 1;
  grdField.ColCount := 1;
  grdField.Cells[0, 0] := '';
  grdData.RowCount := 1;
  grdData.ColCount := 1;
  grdData.Cells[0, 0] := '';
end;

constructor TCnDataSetViewerFrame.Create(AOwner: TComponent);
begin
  inherited;
  FEvaluator := TCnRemoteProcessEvaluator.Create;
end;

destructor TCnDataSetViewerFrame.Destroy;
begin
  FEvaluator.Free;
  inherited;
end;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

procedure TCnDataSetViewerFrame.CloseVisualizer;
begin
  if FOwningForm <> nil then
    FOwningForm.Close;
end;

procedure TCnDataSetViewerFrame.MarkUnavailable(
  Reason: TOTAVisualizerUnavailableReason);
begin
  if Reason = ovurProcessRunning then
    SetAvailableState(asProcRunning)
  else if Reason = ovurOutOfScope then
    SetAvailableState(asOutOfScope);
end;

procedure TCnDataSetViewerFrame.RefreshVisualizer(const Expression, TypeName,
  EvalResult: string);
begin
  AddDataSetContent(Expression, TypeName, EvalResult, CurrentIsCSource);
end;

procedure TCnDataSetViewerFrame.SetClosedCallback(
  ClosedProc: TOTAVisualizerClosedProcedure);
begin
  FClosedProc := ClosedProc;
end;

{$ENDIF}

procedure TCnDataSetViewerFrame.SetForm(AForm: TCustomForm);
begin
  FOwningForm := AForm;
end;

procedure TCnDataSetViewerFrame.SetParent(AParent: TWinControl);
begin
  if AParent = nil then
  begin
{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}
    if Assigned(FClosedProc) then
      FClosedProc;
{$ENDIF}
  end;
  inherited;
end;

{$IFDEF DELPHI120_ATHENS_UP}

procedure TCnDataSetViewerFrame.WMDPIChangedAfterParent(var Message: TMessage);
begin
  inherited;
  if TIDEThemeMetrics.Font.Enabled then
    TIDEThemeMetrics.Font.AdjustDPISize(Font, TIDEThemeMetrics.Font.Size, PixelsPerInch);
end;

{$ENDIF}

procedure TCnDataSetViewerFrame.pcViewsChange(Sender: TObject);
begin
  if pcViews.ActivePage = tsProp then
    mmoProp.SetFocus
  else if pcViews.ActivePage = tsField then
    grdField.SetFocus
  else if pcViews.ActivePage = tsData then
    grdData.SetFocus;
end;

{$IFDEF IDE_HAS_DEBUGGERVISUALIZER}

{ TCnDataSetVisualizerForm }

constructor TCnDataSetVisualizerForm.Create(const Expression: string);
begin
  inherited Create;
  FExpression := Expression;
end;

procedure TCnDataSetVisualizerForm.CustomizePopupMenu(PopupMenu: TPopupMenu);
begin
  // no toolbar
end;

procedure TCnDataSetVisualizerForm.CustomizeToolBar(ToolBar: TToolBar);
begin
 // no toolbar
end;

function TCnDataSetVisualizerForm.EditAction(Action: TEditAction): Boolean;
begin
  Result := False;
end;

procedure TCnDataSetVisualizerForm.FrameCreated(AFrame: TCustomFrame);
begin
  FMyFrame := TCnDataSetViewerFrame(AFrame);
end;

function TCnDataSetVisualizerForm.GetCaption: string;
begin
  Result := Format(SCnDataSetViewerFormCaption, [FExpression]);
end;

function TCnDataSetVisualizerForm.GetEditState: TEditState;
begin
  Result := [];
end;

function TCnDataSetVisualizerForm.GetForm: TCustomForm;
begin
  Result := FMyForm;
end;

function TCnDataSetVisualizerForm.GetFrame: TCustomFrame;
begin
  Result := FMyFrame;
end;

function TCnDataSetVisualizerForm.GetFrameClass: TCustomFrameClass;
begin
  Result := TCnDataSetViewerFrame;
end;

function TCnDataSetVisualizerForm.GetIdentifier: string;
begin
  Result := 'DataSetDebugVisualizer';
end;

function TCnDataSetVisualizerForm.GetMenuActionList: TCustomActionList;
begin
  Result := nil;
end;

function TCnDataSetVisualizerForm.GetMenuImageList: TCustomImageList;
begin
  Result := nil;
end;

function TCnDataSetVisualizerForm.GetToolbarActionList: TCustomActionList;
begin
  Result := nil;
end;

function TCnDataSetVisualizerForm.GetToolbarImageList: TCustomImageList;
begin
  Result := nil;
end;

procedure TCnDataSetVisualizerForm.LoadWindowState(Desktop: TCustomIniFile;
  const Section: string);
begin
  // no desktop saving
end;

procedure TCnDataSetVisualizerForm.SaveWindowState(Desktop: TCustomIniFile;
  const Section: string; IsProject: Boolean);
begin
  // no desktop saving
end;

procedure TCnDataSetVisualizerForm.SetForm(Form: TCustomForm);
begin
  FMyForm := Form;
  if Assigned(FMyFrame) then
    FMyFrame.SetForm(FMyForm);
end;

procedure TCnDataSetVisualizerForm.SetFrame(Frame: TCustomFrame);
begin
   FMyFrame := TCnDataSetViewerFrame(Frame);
end;

{$ENDIF}

procedure ShowDataSetExternalViewer(const Expression: string);
var
  F: TCnIdeDockForm;
  Fm: TCnDataSetViewerFrame;
begin
  if not CnWizDebuggerObjectInheritsFrom(Expression, 'TDataSet') then
  begin
    ErrorDlg(Format(SCnDebugErrorExprNotAClass, [Expression, 'TDataSet']));
    Exit;
  end;

  F := TCnIdeDockForm.Create(Application);
  F.Caption := Format(SCnDataSetViewerFormCaption, [Expression]);
  Fm := TCnDataSetViewerFrame.Create(F);
  Fm.SetForm(F);
  Fm.Parent := F;
  Fm.Align := alClient;
  Fm.AddDataSetContent(Expression, '', '', CurrentIsCSource);

  F.Show;
end;

end.

