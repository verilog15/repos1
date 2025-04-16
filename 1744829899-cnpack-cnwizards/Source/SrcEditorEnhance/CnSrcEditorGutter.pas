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

unit CnSrcEditorGutter;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�����༭���к���ʾ��Ԫ
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
*           �ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2017.11.04
*               BDS ��Ҳģ�������������϶�ѡ��Ĺ��ܣ�������ʮ�л���ģʽ
*           2016.06.17
*               �����߸Ķ����Ļ��ƣ�Ȼ�����޷���ȡ�Ķ����кţ�ֻ������
*           2004.12.25
*               ������Ԫ����ԭ CnSrcEditorEnhancements �Ƴ�
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNSRCEDITORENHANCE}

uses
  Windows, Messages, Classes, Graphics, SysUtils, Controls, Menus, Forms, ToolsAPI,
  IniFiles, ExtCtrls, CnWizShareImages, CnWizMenuAction,
  CnEditControlWrapper, CnWizNotifier, CnIni, CnPopupMenu, CnFastList, CnEventBus;

type
  TCnSrcEditorGutter = class;

  TCnIdentReceiver = class(TInterfacedObject, ICnEventBusReceiver)
  private
    FTimer: TTimer;
    FGutter: TCnSrcEditorGutter;
    procedure TimerOnTimer(Sender: TObject);
  public
    constructor Create(AGutter: TCnSrcEditorGutter);
    destructor Destroy; override;

    procedure OnEvent(Event: TCnEvent);
  end;

{ TCnSrcEditorGutter }

  TCnSrcEditorGutterMgr = class;
  TCnGutterWidth = 1..10;

  TCnSrcEditorGutter = class(TCustomControl)
  private
    FActive: Boolean;
    FPainting: Boolean;
    FMouseDown: Boolean;
    FDragging: Boolean;
    FStartLine: Integer;
    FEndLine: Integer;
    FGutterMgr: TCnSrcEditorGutterMgr;
    FEditControl: TControl;
    FEditWindow: TCustomForm;
    FLineHeight: Integer;
    FMenu: TPopupMenu;
    FLinesReceiver: ICnEventBusReceiver;
    FIdentLines: TCnList;
    FIdentCols: TCnList;
    FSelectLineTimer: TTimer;
{$IFDEF BDS}
    FIDELineNumMenu: TMenuItem;
{$ENDIF}
    FOldLineWidth: Integer;
    FPosInfo: TCnEditControlInfo;
{$IFNDEF BDS}
    FLineInfo: TCnList;
{$ENDIF}
    procedure MenuPopup(Sender: TObject);
    procedure EditorChanged(Editor: TCnEditorObject; ChangeType: TCnEditorChangeTypes);
    procedure OnClearBookMarks(Sender: TObject);
    procedure OnEnhConfig(Sender: TObject);
    procedure OnGotoLine(Sender: TObject);
    procedure OnLineClose(Sender: TObject);
{$IFDEF BDS}
    procedure OnShowIDELineNum(Sender: TObject);
{$ENDIF}
    procedure SubItemClick(Sender: TObject);
    function GetLineCountRect: TRect;
    function MapYToLine(Y: Integer; EditView: IOTAEditView = nil): Integer;

    // ���ַ����Ҳ����� LinesList �е�ƥ����±꣬-1��ʾû��ƥ�䡣
    function MatchPointToLineArea(LinesList: TCnList; const Delta, Y, AreaHeight,
      TotalLine: Integer): Integer;
    procedure ToggleBookmark;
    procedure SetEditControl(const Value: TControl);
  protected
{$IFDEF BDS}
    procedure SetEnabled(Value: Boolean); override;
{$ENDIF}
    procedure DblClick; override;
    function CheckPosNavigate: Boolean;
    procedure InitPopupMenu;
    procedure Paint; override;

    procedure MouseDown(Button: TMouseButton; Shift: TShiftState;
      X, Y: Integer); override;
    procedure MouseMove(Shift: TShiftState; X, Y: Integer); override;
    procedure MouseUp(Button: TMouseButton; Shift: TShiftState;
      X, Y: Integer); override;
    procedure Click; override;
    procedure OnSelectLineTimer(Sender: TObject);
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure UpdateStatus;
    procedure UpdateStatusOnIdle(Sender: TObject);
    procedure LanguageChanged(Sender: TObject);
    property EditControl: TControl read FEditControl write SetEditControl;
    property EditWindow: TCustomForm read FEditWindow write FEditWindow;
    property OldLineWidth: Integer read FOldLineWidth write FOldLineWidth;
    property Menu: TPopupMenu read FMenu;
  end;

{ TCnSrcEditorGutterMgr }

  TCnSrcEditorGutterMgr = class(TPersistent)
  private
    FList: TList;
    FFont: TFont;
    FShowLineNumber: Boolean;
    FOnEnhConfig: TNotifyEvent;
    FActive: Boolean;
    FCurrFont: TFont;
    FShowLineCount: Boolean;
    FAutoWidth: Boolean;
    FMinWidth: TCnGutterWidth;
    FFixedWidth: TCnGutterWidth;
    FShowModifier: Boolean;
    FDblClickToggleBookmark: Boolean;
    FClickSelectLine: Boolean;
    FDragSelectLines: Boolean;
    FTenMode: Boolean;
    FRelativeNumber: Boolean;
    procedure SetFont(const Value: TFont);
    procedure SetShowLineNumber(const Value: Boolean);
    procedure SetActive(const Value: Boolean);
    function GetCount: Integer;
    function GetGutters(Index: Integer): TCnSrcEditorGutter;
    procedure SetCurrFont(const Value: TFont);
    procedure EditControlNotify(EditControl: TControl; EditWindow: TCustomForm;
      Operation: TOperation);
    procedure SetFixedWidth(const Value: TCnGutterWidth);
    procedure SetMinWidth(const Value: TCnGutterWidth);
    procedure SetShowModifier(const Value: Boolean);
    procedure SetTenMode(const Value: Boolean);
    procedure SetRelativeNumber(const Value: Boolean);
  protected
    procedure ThemeChanged(Sender: TObject);
    procedure DoUpdateGutters(EditWindow: TCustomForm; EditControl: TControl; Context:
      Pointer);
    procedure DoEnhConfig;
    procedure DoLineClose;
    function CanShowGutter: Boolean;
  public
    constructor Create;
    destructor Destroy; override;

    procedure UpdateGutters;
    procedure LanguageChanged(Sender: TObject);
    procedure LoadSettings(Ini: TCustomIniFile);
    procedure SaveSettings(Ini: TCustomIniFile);
    procedure ResetSettings(Ini: TCustomIniFile);

    property Count: Integer read GetCount;
    property Gutters[Index: Integer]: TCnSrcEditorGutter read GetGutters;

    property Font: TFont read FFont write SetFont;
    property CurrFont: TFont read FCurrFont write SetCurrFont;
  published
    property ShowLineNumber: Boolean read FShowLineNumber write SetShowLineNumber;
    {* �Ƿ��ڱ༭�������ʾ�к���}
    property ShowLineCount: Boolean read FShowLineCount write FShowLineCount;
    {* �Ƿ����к����·���ʾ������}
    property AutoWidth: Boolean read FAutoWidth write FAutoWidth;
    {* �Ƿ�����к�λ���Զ������к�����ȣ����ʹ���к����̶����}
    property MinWidth: TCnGutterWidth read FMinWidth write SetMinWidth;
    {* �Ƿ��к�����С���}
    property FixedWidth: TCnGutterWidth read FFixedWidth write SetFixedWidth;
    {* �к����̶����}

    property ClickSelectLine: Boolean read FClickSelectLine write FClickSelectLine;
    {* �Ƿ񵥻�ѡ������}
    property DragSelectLines: Boolean read FDragSelectLines write FDragSelectLines;
    {* �Ƿ��϶�ѡ����У�BDS ���Ѿ��д˹����˵�Ҳ����}
    property DblClickToggleBookmark: Boolean read FDblClickToggleBookmark write FDblClickToggleBookmark;
    {* �Ƿ�˫���л���ǩ}
    property TenMode: Boolean read FTenMode write SetTenMode;
    {* �Ƿ�ʹ������ģʽ��ֻ��ʾ��ʮ����}
    property RelativeNumber: Boolean read FRelativeNumber write SetRelativeNumber;
    {* �Ƿ�ʹ���������}

    property Active: Boolean read FActive write SetActive;
    property ShowModifier: Boolean read FShowModifier write SetShowModifier;
    property OnEnhConfig: TNotifyEvent read FOnEnhConfig write FOnEnhConfig;
  end;

{$ENDIF CNWIZARDS_CNSRCEDITORENHANCE}

implementation

{$IFDEF CNWIZARDS_CNSRCEDITORENHANCE}

uses
{$IFDEF DEBUG}
  CnDebug,
{$ENDIF}
  CnCommon, CnWizClasses, CnWizManager, CnBookmarkWizard, CnWizUtils, CnWizConsts,
  CnWizIdeUtils, Math;

const
  SCnSrcEditorGutter = 'CnSrcEditorGutter';
  csIDEShowLineNumbers = 'ShowLineNumbers';

  csBevelWidth = 4;
  csModifierWidth = 5;

  csIdentLineHeightDrawDelta = 1;
  csIdentLineHeightMouseDelta = 1;
  csIdentLineRightMargin = 3;
  csIdentLineShowDelay = 200; // 200ms for show delay
  csMaxLineLength = 2048;

  csDefaultModifiedColor = clYellow;
  csDefaultModifiedSavedColor = clGreen;
  csTotalLineBackgroundColor = clNavy;
  csTotalLineNumberColor = clWhite;

  csDefaultLineNumberColor = clNavy;
  csDefaultCurrentLineNumberColor = clRed;
  csDarkTotalLineNumberBackgroundColor = clNavy;
  csDarkTotalLineNumberColor = clYellow;

  csGutter = 'Gutter';
{$IFDEF BDS}
  csShowLineNumber = 'ShowLineNumberBDS';
  csShowLineNumberDef = True;
  csShowModifier = 'ShowModifierBDS';
  csShowModifierDef = False; // BDS ���Ѿ����޸Ĺ�������ʾ�ˣ�����
{$ELSE}
  csShowLineNumber = 'ShowLineNumber';
  csShowLineNumberDef = True;
  csShowModifier = 'ShowModifier';
  csShowModifierDef = False; // D5��6��7 ���޷���ȡ�޸ĵ��У�ֻ������Ϊ False;
{$ENDIF}
  csFont = 'Font';
  csCurrFont = 'CurrFont';
  csShowLineCount = 'ShowLineCount';
  csAutoWidth = 'AutoWidth';
  csMinWidth = 'MinWidth';
  csFixedWidth = 'FixedWidth';

  csClickSelectLine = 'ClickSelectLine';
  csDragSelectLines = 'DragSelectLines';
  csTenMode = 'TenMode';
  csRelativeNumber = 'RelativeNumber';
  csDblClickToggleBookmark = 'DblClickToggleBookmark';

  CN_GUTTER_LINE_MODIFIER_CHANGED = 1;
  CN_GUTTER_LINE_MODIFIER_SAVED = 2;

{ TCnSrcEditorGutter }

constructor TCnSrcEditorGutter.Create(AOwner: TComponent);
begin
  inherited;
  BevelOuter := bvLowered;
  BevelInner := bvRaised;
  Width := 10;
  Align := alLeft;
  DoubleBuffered := True;

  // ����ģʽ���ð���ɫ
  if CnThemeWrapper.CurrentIsDark then
    Color := csDarkBackgroundColor;

  FMenu := TPopupMenu.Create(Self);
  FMenu.Images := dmCnSharedImages.GetMixedImageList;
  FMenu.OnPopup := MenuPopup;
  InitPopupMenu;
  PopupMenu := FMenu;

  FIdentLines := TCnList.Create;
  FIdentCols := TCnList.Create;
  FLinesReceiver := TCnIdentReceiver.Create(Self);
  EventBus.RegisterReceiver(FLinesReceiver, EVENT_HIGHLIGHT_IDENT_POSITION);
  FSelectLineTimer := TTimer.Create(Self);
  FSelectLineTimer.Enabled := False;
  FSelectLineTimer.Interval := 200;
  FSelectLineTimer.OnTimer := OnSelectLineTimer;

{$IFNDEF BDS}
  FLineInfo := TCnList.Create;
{$ENDIF}
  EditControlWrapper.AddEditorChangeNotifier(EditorChanged);
end;

destructor TCnSrcEditorGutter.Destroy;
begin
  EventBus.UnRegisterReceiver(FLinesReceiver);
  FLinesReceiver := nil;

  FGutterMgr.FList.Remove(Self);
  EditControlWrapper.RemoveEditorChangeNotifier(EditorChanged);
  FIdentLines.Free;
  FIdentCols.Free;
{$IFNDEF BDS}
  FLineInfo.Free;
{$ENDIF}
  inherited;
end;

//------------------------------------------------------------------------------
// ״̬����
//------------------------------------------------------------------------------

{$IFDEF BDS}
procedure TCnSrcEditorGutter.SetEnabled(Value: Boolean);
begin
// ʲôҲ���������赲 BDS ���л�ҳ��ʱ Disable �������Ĳ���
end;
{$ENDIF}

procedure TCnSrcEditorGutter.EditorChanged(Editor: TCnEditorObject;
  ChangeType: TCnEditorChangeTypes);
var
  OldHeight: Integer;
begin
  if FGutterMgr.CanShowGutter and (Editor.EditControl = FEditControl) then
  begin
    if ChangeType * [ctView, ctTopEditorChanged] <> [] then
      CnWizNotifierServices.ExecuteOnApplicationIdle(UpdateStatusOnIdle)
    else if ChangeType * [ctWindow, ctCurrLine, ctFont, ctVScroll
      {$IFDEF IDE_EDITOR_ELIDE}, ctElided, ctUnElided {$ENDIF}] <> [] then
      Repaint;

    if ChangeType * [ctView, ctFont, ctOptionChanged] <> [] then
    begin
{$IFDEF DEBUG}
      CnDebugger.LogMsg('SrcEditorGutter.EditorChanged for View/Font/Option. Update Line Height');
{$ENDIF}
      OldHeight := FLineHeight;
      FLineHeight := EditControlWrapper.GetEditControlCharHeight(FEditControl);
      if OldHeight <> FLineHeight then
        Repaint;
    end;
  end;
end;

procedure TCnSrcEditorGutter.UpdateStatus;
var
  EditorObj: TCnEditorObject;
  MaxRowWidth, MaxCurRowWidth: Integer;
  EndLine: Integer;

  function GetNumWidth(Len: Integer): Integer;
  begin
    Result := Canvas.TextWidth(IntToStrEx(0, Len));
  end;

begin
  if not ClassNameIs('TCnSrcEditorGutter') then // �ǳ���ֵĲ��������ӵĻ��ݱ�������� D7 �������ĳЩ TEdit �� Visible ����仯
    Exit;

  if not FActive then
  begin
    Visible := False;
    Exit;
  end;

  EditorObj := EditControlWrapper.GetEditorObject(EditControl);
  if (EditorObj = nil) or not EditorObj.EditorIsOnTop then
  begin
    Visible := False;
    Exit;
  end;

  Visible := True;
  Canvas.Font := Font;

  if FGutterMgr.AutoWidth then
  begin
    FPosInfo := EditControlWrapper.GetEditControlInfo(EditControl);
    if FGutterMgr.ShowLineCount then
      EndLine := FPosInfo.LineCount
    else
      EndLine := EditorObj.ViewBottomLine;
    MaxRowWidth := Max(GetNumWidth(Length(IntToStr(EndLine))),
      GetNumWidth(FGutterMgr.MinWidth));

    Canvas.Font := FGutterMgr.CurrFont;
    MaxCurRowWidth := Max(GetNumWidth(Length(IntToStr(EndLine))),
      GetNumWidth(FGutterMgr.MinWidth));
  end
  else
  begin
    MaxRowWidth := GetNumWidth(FGutterMgr.FixedWidth);
    Canvas.Font := FGutterMgr.CurrFont;
    MaxCurRowWidth := GetNumWidth(FGutterMgr.FixedWidth);
  end;

  Canvas.Font := Font;
  MaxRowWidth := Max(MaxRowWidth, MaxCurRowWidth);
  // ��ǰ��������ܸ�����û��Ʋ��㣬ȡ����
  Width := MaxRowWidth + csBevelWidth + 1; // ���һ���
  if FGutterMgr.ShowModifier then
    Width := Width + csModifierWidth;

  Invalidate;
end;

procedure TCnSrcEditorGutter.UpdateStatusOnIdle(Sender: TObject);
begin
  UpdateStatus;
end;

function TCnSrcEditorGutter.GetLineCountRect: TRect;
begin
  Canvas.Font := FGutterMgr.Font;
  Result := Rect(1, ClientHeight - Canvas.TextHeight(IntToStr(FPosInfo.LineCount)),
    Width - csBevelWidth, ClientHeight);
end;

procedure TCnSrcEditorGutter.Paint;
var
  R: TRect;
  StrNum: string;
  I, X, Y, Idx, MaxRow: Integer;
  EditorObj: TCnEditorObject;
  OldColor: TColor;
begin
  if FPainting or not Visible or (EditControl = nil) or (Parent = nil) then
    Exit;

  FPainting := True;
  try
    if (FIdentLines.Count > 1) and (GetCurrentEditControl = EditControl) then // ��λ������ͼֻ���ж��������ǰʱ��
    begin
      MaxRow := FPosInfo.LineCount;
      if MaxRow <= 0 then
        Exit;

      Canvas.Pen.Style := psClear;

      OldColor := Canvas.Brush.Color;
      // ����ģʽ�µı���ɫ�� Color ���Ծ��������ڴ˴��޸�
      Canvas.Brush.Color := $00B0B0B0;
      Canvas.Brush.Style := bsSolid;

      for I := 0 to FIdentLines.Count - 1 do
      begin
        Y := Height * Integer(FIdentLines[I]) div MaxRow;
        R := Rect(0, Y - csIdentLineHeightDrawDelta, Width - csIdentLineRightMargin,
          Y + csIdentLineHeightDrawDelta);
        Canvas.FillRect(R);
      end;

      Canvas.Brush.Color := OldColor;
    end;

    if FLineHeight > 0 then
    begin
      EditorObj := EditControlWrapper.GetEditorObject(EditControl);
      if EditorObj = nil then Exit;
      FPosInfo := EditControlWrapper.GetEditControlInfo(EditControl);

      if FGutterMgr.AutoWidth then
      begin
        if FGutterMgr.ShowLineCount then
          MaxRow := FPosInfo.LineCount
        else
          MaxRow := EditorObj.ViewBottomLine;

        if (OldLineWidth <> Length(IntToStr(MaxRow))) then
        begin
          OldLineWidth := Length(IntToStr(MaxRow));
          UpdateStatus;
          Exit;
        end;
      end;

      for I := 0 to EditorObj.ViewLineCount - 1 do
      begin
        Idx := EditorObj.ViewLineNumber[I];
        if Idx <> FPosInfo.CaretY then
        begin
          Canvas.Font := FGutterMgr.Font;
          // ����ģʽ�µ���Ĭ��������ɫ
          if CnThemeWrapper.CurrentIsDark and (Canvas.Font.Color = csDefaultLineNumberColor) then
            Canvas.Font.Color := csDarkFontColor;
        end
        else
        begin
          Canvas.Font := FGutterMgr.CurrFont;
        end;

        Canvas.Brush.Style := bsClear;
        R := Rect(1, I * FLineHeight, Width - csBevelWidth, (I + 1) * FLineHeight);
        if FGutterMgr.ShowModifier then
          R.Right := R.Right - csModifierWidth;

        if not FGutterMgr.TenMode or (Idx mod 10 = 0)
          or (Idx = FPosInfo.CaretY) or (Idx = 1) then // ��һ��Ҳ��
        begin
          if FGutterMgr.RelativeNumber and not (Idx = FPosInfo.CaretY) then
            StrNum := IntToStr(Abs(Idx - FPosInfo.CaretY))
          else
            StrNum := IntToStr(Idx);

          DrawText(Canvas.Handle, PChar(StrNum), Length(StrNum), R, DT_VCENTER or
            DT_RIGHT or DT_SINGLELINE);
        end
        else // ��ʮģʽ�£�������ͨ�̵������������
        begin
          Canvas.Pen.Width := 1;
          Canvas.Pen.Style := psSolid;
          Canvas.Pen.Color := FGutterMgr.Font.Color;

          Y := (R.Top + R.Bottom) div 2;
          if Idx mod 5 = 0 then
          begin
            X := R.Right - 8;
            Canvas.MoveTo(X, Y);
            Canvas.LineTo(X + 6, Y);
          end
          else
          begin
            X := R.Right - 4;
            Ellipse(Canvas.Handle, X - 1, Y - 1, X + 1, Y + 1);
          end;
        end;

        if FGutterMgr.ShowModifier then
        begin
          OldColor := Canvas.Brush.Color;
          Canvas.Brush.Color := csDefaultModifiedColor;
          R.Left := Width - csBevelWidth - csModifierWidth + 2;
          R.Right := Width - csBevelWidth;
          Canvas.FillRect(R);
          Canvas.Brush.Color := OldColor;
        end;
      end;

      // ��������
      if FGutterMgr.ShowLineCount then
      begin
        R := GetLineCountRect;
        Canvas.Font := FGutterMgr.Font;
        if CnThemeWrapper.CurrentIsDark then
        begin
          Canvas.Font.Color := csDarkTotalLineNumberColor;
          Canvas.Brush.Color := csDarkTotalLineNumberBackgroundColor;
        end
        else
        begin
          Canvas.Font.Color := csTotalLineNumberColor;
          Canvas.Brush.Color := csTotalLineBackgroundColor;
        end;

        StrNum := IntToStr(FPosInfo.LineCount);
        if FGutterMgr.ShowModifier then
          R.Right := R.Right - csModifierWidth;

        Canvas.FillRect(R);
        DrawText(Canvas.Handle, PChar(StrNum), Length(StrNum), R, DT_VCENTER or
          DT_RIGHT or DT_SINGLELINE);
      end;
    end;

    Canvas.Pen.Width := 1;
    Canvas.Pen.Style := psSolid;
    Canvas.Pen.Color := clBtnHighlight;
    Canvas.MoveTo(Width - 1, 0);
    Canvas.LineTo(Width - 1, Height);

    Canvas.Pen.Color := clBtnShadow;
    Canvas.MoveTo(Width - 2, 0);
    Canvas.LineTo(Width - 2, Height);
  finally
    FPainting := False;
  end;
end;

//------------------------------------------------------------------------------
// ����¼�
//------------------------------------------------------------------------------

procedure TCnSrcEditorGutter.ToggleBookmark;
var
  EditView: IOTAEditView;
  Row: Integer;
  ID: Integer;
  EditPos, SavePos: TOTAEditPos;
  Pt: TPoint;

  function GetBlankBookmarkID: Integer;
  var
    I: Integer;
  begin
    Result := 1;
    for I := 1 to 10 do
      if EditView.BookmarkPos[I mod 10].Line = 0 then
      begin
        Result := I mod 10;
        Exit;
      end;
  end;

  function FindBookmark(Row: Integer): Integer;
  var
    I: Integer;
  begin
    Result:= -1;
    for I := 0 to 9 do
    begin
      if EditView.BookmarkPos[I].Line = Row then
      begin
        Result := I;
        Exit;
      end;
    end;
  end;

begin
  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    Pt := Mouse.CursorPos;
    Pt := ScreenToClient(Pt);
    Row := MapYToLine(Pt.y, EditView);

    SavePos := EditView.CursorPos;
    EditPos := EditView.CursorPos;
    EditPos.Line := Row;
    EditView.CursorPos := EditPos;

    ID := FindBookmark(Row);
    if ID = -1 then
      ID := GetBlankBookmarkID;
    EditView.BookmarkToggle(ID);
    EditView.CursorPos := SavePos;
    EditView.Paint;
  end;
end;

procedure TCnSrcEditorGutter.DblClick;
var
  EditView: IOTAEditView;
  Pt: TPoint;
begin
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnSrcEditorGutter.DoubleClick. Cancel SelectLine Timer.');
{$ENDIF}

  FSelectLineTimer.Enabled := False;
  if not FGutterMgr.DblClickToggleBookmark then
    Exit;

  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    Pt := Mouse.CursorPos;
    Pt := ScreenToClient(Pt);

    if FGutterMgr.ShowLineCount and PtInRect(GetLineCountRect, Pt) then
    begin
      OnGotoLine(Self);
    end
    else
    begin
      ToggleBookmark;
    end;
  end;
end;

function TCnSrcEditorGutter.CheckPosNavigate: Boolean;
var
  EditView: IOTAEditView;
  Pt: TPoint;
  LineIdx: Integer;
begin
  Result := False;
  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    Pt := Mouse.CursorPos;
    Pt := ScreenToClient(Pt);

    LineIdx := MatchPointToLineArea(FIdentLines, csIdentLineHeightMouseDelta, Pt.y, Height,
      FPosInfo.LineCount);

    if LineIdx >= 0 then
    begin
      Result := True;
      EditView.Position.Move(Integer(FIdentLines[LineIdx]), Integer(FIdentCols[LineIdx]));
      EditView.MoveViewToCursor;
      EditView.Paint;
    end;
  end;
end;

procedure TCnSrcEditorGutter.LanguageChanged(Sender: TObject);
begin
  InitPopupMenu;
end;

//------------------------------------------------------------------------------
// �˵����
//------------------------------------------------------------------------------

procedure TCnSrcEditorGutter.InitPopupMenu;
var
  Wizard: TCnBaseWizard;
  Item: TMenuItem;
begin
  Menu.Items.Clear;
  Wizard := CnWizardMgr.WizardByClassName('TCnBookmarkWizard');
  if Wizard <> nil then
  begin
    Item := TMenuItem.Create(Menu);
    Menu.Items.Add(Item);
    Item.Action := TCnMenuWizard(Wizard).Action;
  end;
  Item := AddMenuItem(Menu.Items, SCnLineNumberClearBookMarks, OnClearBookMarks);
  Item.ImageIndex := dmCnSharedImages.CalcMixedImageIndex(31);
  AddMenuItem(Menu.Items, SCnLineNumberGotoLine, OnGotoLine);
  AddSepMenuItem(Menu.Items);

  Item := AddMenuItem(Menu.Items, SCnEditorEnhanceConfig, OnEnhConfig);
  Item.ImageIndex := CnWizardMgr.ImageIndexByWizardClassNameAndCommand('TCnIdeEnhanceMenuWizard',
    SCnIdeEnhanceMenuCommand + 'TCnSrcEditorEnhance');

  AddSepMenuItem(Menu.Items);
{$IFDEF BDS}
  FIDELineNumMenu := AddMenuItem(Menu.Items, SCnLineNumberShowIDELineNum, OnShowIDELineNum);
{$ENDIF}
  Item := AddMenuItem(Menu.Items, SCnLineNumberClose, OnLineClose);
  Item.ImageIndex := dmCnSharedImages.CalcMixedImageIndex(13);
end;

procedure TCnSrcEditorGutter.MenuPopup(Sender: TObject);
var
  EditView: IOTAEditView;
  Item, SubItem: TMenuItem;
  I: Integer;
{$IFDEF BDS}
  Options: IOTAEnvironmentOptions;
{$ENDIF}
begin
  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    if Menu.Items.Items[0].Tag <> 1 then // �� Tag Ϊ 1 ����ʶ��ǩ�б�˵���
    begin
      Item := TMenuItem.Create(Menu);
      Item.Caption := SCnLineNumberGotoBookMark;
      Item.Tag := 1;
      Menu.Items.Insert(0, Item);
    end
    else
      Item := Menu.Items.Items[0];

    Item.Clear;
    for I := 0 to 9 do
    begin
      SubItem := TMenuItem.Create(Menu);
      SubItem.Caption := Format('Bookmark &%d', [I]);
      SubItem.Tag := I;
      SubItem.OnClick := SubItemClick;
      Item.Add(SubItem);
      SubItem.Enabled := EditView.BookmarkPos[I].Line > 0
    end;
  end;

{$IFDEF BDS}
  Options := CnOtaGetEnvironmentOptions;
  if Options <> nil then
  begin
    FIDELineNumMenu.Checked := Options.GetOptionValue(csIDEShowLineNumbers);
    FIDELineNumMenu.Visible := True;
  end
  else
    FIDELineNumMenu.Visible := False;
{$ENDIF}
end;

procedure TCnSrcEditorGutter.OnClearBookMarks(Sender: TObject);
var
  EditView: IOTAEditView;
  EditPos, SavePos: TOTAEditPos;
  I: Integer;
begin
  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    SavePos := EditView.CursorPos;
    for I := 0 to 9 do
    begin
      if EditView.BookmarkPos[I].Line > 0 then
      begin
        EditPos := EditView.CursorPos;
        EditPos.Line := EditView.BookmarkPos[I].Line;
        EditView.CursorPos := EditPos;
        EditView.BookmarkToggle(I);
      end;
    end;
    EditView.CursorPos := SavePos;
    EditView.Paint;
  end;
end;

procedure TCnSrcEditorGutter.OnEnhConfig(Sender: TObject);
begin
  FGutterMgr.DoEnhConfig;
end;

procedure TCnSrcEditorGutter.OnGotoLine(Sender: TObject);
var
  EditView: IOTAEditView;
begin
  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    EditView.Position.GotoLine(0);
    EditView.Paint;
  end;
end;

{$IFDEF BDS}
procedure TCnSrcEditorGutter.OnShowIDELineNum(Sender: TObject);
var
  AShow: Boolean;
  Options: IOTAEnvironmentOptions;
  I: Integer;
begin
  Options := CnOtaGetEnvironmentOptions;
  if Options <> nil then
  begin
    AShow := Options.GetOptionValue(csIDEShowLineNumbers);
    Options.SetOptionValue(csIDEShowLineNumbers, not AShow);
    for I := 0 to EditControlWrapper.EditorCount - 1 do
      EditControlWrapper.Editors[I].NotifyIDEGutterChanged;
  end;
end;
{$ENDIF}

procedure TCnSrcEditorGutter.OnLineClose(Sender: TObject);
begin
  FGutterMgr.DoLineClose;
end;

procedure TCnSrcEditorGutter.SubItemClick(Sender: TObject);
var
  EditView: IOTAEditView;
  ID: Integer;
  EditPos: TOTAEditPos;
begin
  EditView := EditControlWrapper.GetEditView(EditControl);
  if Assigned(EditView) then
  begin
    ID := (Sender as TComponent).Tag;
    if ID in [0..9] then
    begin
      EditView.BookmarkGoto(ID);
      EditPos.Line := EditView.BookmarkPos[ID].Line;
      if EditPos.Line > 0 then
      begin
        EditPos.Col := 1;
        EditView.CursorPos := EditPos;
        EditView.MoveViewToCursor;
        EditView.Paint;
        if (EditControl <> nil) and (EditControl is TWinControl) and EditControl.Visible then
          (EditControl as TWinControl).SetFocus;
      end;
    end;
  end;
end;

function TCnSrcEditorGutter.MatchPointToLineArea(LinesList: TCnList; const Delta,
  Y, AreaHeight, TotalLine: Integer): Integer;
var
  I, J, M: Integer;

  // Y λ�ú�ĳ������Ƚϣ����ڡ�С���� 1 ��0��-1
  function ComparePointWithLinePos(LineNo: Integer): Integer;
  var
    Up, Down, YPos: Integer;
  begin
    YPos := (LineNo * AreaHeight) div TotalLine;

    Up := YPos - Delta;
    if Y < Up then
    begin
      Result := -1;
      Exit;
    end;

    Down := YPos + Delta;
    if Y <= Down then
    begin
      Result := 0;
      Exit;
    end;

    Result := 1;
  end;

begin
  Result := -1;
  if (LinesList = nil) or (LinesList.Count = 0) or (TotalLine = 0) then
    Exit;

  I := 0;
  J := LinesList.Count - 1;

  while I <= J do
  begin
    M := (I + J) div 2;
    case ComparePointWithLinePos(Integer(LinesList[M])) of
    1:
      begin
        // Y �����������
        I := M + 1;
      end;
    0:
      begin
        Result := M;
        Exit;
      end;
    -1:
      begin
        // Y ����������С
        J := M - 1;
      end;
    end;
  end;
end;

procedure TCnSrcEditorGutter.MouseMove(Shift: TShiftState; X, Y: Integer);
var
  Idx: Integer;
  Line: Integer;
  FirstDrag: Boolean;
  View: IOTAEditView;
  Block: IOTAEditBlock;
  Position: IOTAEditPosition;
begin
  inherited;
{$IFDEF DEBUG}
//  CnDebugger.LogMsg('TCnSrcEditorGutter.MouseMove: Dragging ' + IntToStr(Integer(FDragging)));
{$ENDIF}

  if FMouseDown then
  begin
    FirstDrag := not FDragging;
    FDragging := True;

    if FGutterMgr.DragSelectLines then // BDS ��Ҳ�����϶�ѡ��
    begin
      // ѡ������
      Line := MapYToLine(Y);
      if Line <> FEndLine then
      begin
        FEndLine := Line;
        if FEndLine >= 0 then
        begin
{$IFDEF DEBUG}
          CnDebugger.LogFmt('TCnSrcEditorGutter.MouseMove: Select From Line %d to %d.',
            [FStartLine, FEndLine]);
{$ENDIF}

          View := CnOtaGetTopMostEditView;
          if View = nil then
            Exit;

          Position := View.Position;
          Block := View.Block;
          if FirstDrag then
          begin
            // �տ�ʼ�϶������¿�ʼѡ��
            CnOtaMoveAndSelectLine(FStartLine, View);
            Position.Move(FStartLine, 1);
            Position.MoveBOL;
            View.Paint;
          end
          else
          begin
            // �Ѿ���ʼ�϶�ѡ���ˣ���չѡ����
            if FEndLine > FStartLine then
            begin
              Block.Extend(FEndLine + 1, 1);
              //View.MoveViewToCursor;
            end
            else if FEndLine < FStartLine then
            begin
              Block.Extend(FEndLine, 1);
              //View.MoveViewToCursor;
            end;
{$IFDEF BDS}
            View.Paint;
{$ENDIF}
          end;
        end;
      end;
    end;

    Exit;
  end;

  Idx := MatchPointToLineArea(FIdentLines, csIdentLineHeightMouseDelta, Y, Height,
    FPosInfo.LineCount);
  if Idx >= 0 then
  begin
    Cursor := crHandPoint;
    Hint := IntToStr(Integer(FIdentLines[Idx]));
    ShowHint := True;
  end
  else
  begin
    Cursor := crDefault;
    Hint := '';
    ShowHint := False;
    Application.CancelHint;
  end;
end;

procedure TCnSrcEditorGutter.MouseDown(Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
  Pt: TPoint;
begin
  inherited;
  if Button = mbLeft then
  begin
{$IFDEF DEBUG}
    CnDebugger.LogMsg('TCnSrcEditorGutter.MouseDown');
{$ENDIF}

    Pt.x := X;
    Pt.y := Y;
    if FGutterMgr.ShowLineCount and PtInRect(GetLineCountRect, Pt) then
      Exit;

    FMouseDown := True;
    FDragging := False;
    FEndLine := -1; // ���������

    // ��¼�еĿ�ʼλ�ù��϶�ѡ����
    FStartLine := MapYToLine(Y);
  end;
end;

procedure TCnSrcEditorGutter.MouseUp(Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  inherited;
  if Button = mbLeft then
  begin
{$IFDEF DEBUG}
    CnDebugger.LogMsg('TCnSrcEditorGutter.MouseUp');
{$ENDIF}

    FMouseDown := False;
    FEndLine := -1; // ���������
    FDragging := False;
  end;
end;

procedure TCnSrcEditorGutter.Click;
begin
  inherited;
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnSrcEditorGutter.Click');
{$ENDIF}

  if not FDragging then
  begin
    if not CheckPosNavigate then
    begin
      // ѡ������
      if FGutterMgr.ClickSelectLine and (FStartLine > -1) then
      begin
{$IFDEF DEBUG}
        CnDebugger.LogMsg('TCnSrcEditorGutter.Click. May Start Timer.');
{$ENDIF}
        if FGutterMgr.DblClickToggleBookmark then // ��˫���л���ǩʱҪ��ʱһС���
          FSelectLineTimer.Enabled := True
        else
          CnOtaMoveAndSelectLine(FStartLine);
      end
      else if not FGutterMgr.ClickSelectLine then // ������ѡ����ʱ��ʹ�õ����л���ǩ
        ToggleBookmark;
    end;
  end;
end;

function TCnSrcEditorGutter.MapYToLine(Y: Integer; EditView: IOTAEditView): Integer;
var
  P: TPoint;
begin
  P.x := 0;
  P.y := Y;
  Result := EditControlWrapper.GetLineFromPoint(P, EditControl, EditView);
end;

procedure TCnSrcEditorGutter.OnSelectLineTimer(Sender: TObject);
begin
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnSrcEditorGutter.OnSelectLineTimer. Timer Triggered to Select Line.');
{$ENDIF}

  if FGutterMgr.ClickSelectLine and (FStartLine > -1) then
    CnOtaMoveAndSelectLine(FStartLine);
  FSelectLineTimer.Enabled := False;
end;

procedure TCnSrcEditorGutter.SetEditControl(const Value: TControl);
begin
  if FEditControl <> Value then
  begin
    FEditControl := Value;
    FLineHeight := EditControlWrapper.GetEditControlCharHeight(FEditControl);
  end;
end;

{ TCnSrcEditorGutterMgr }

constructor TCnSrcEditorGutterMgr.Create;
var
  Option: IOTAEditOptions;
begin
  inherited;
  FList := TList.Create;
  FFont := TFont.Create;
  FCurrFont := TFont.Create;

  Option := CnOtaGetEditOptions;
  if Option <> nil then
  begin
    FFont.Name := Option.FontName;
    FFont.Size := Option.FontSize;
  end
  else
  begin
    FFont.Name := 'Verdana';
    FFont.Size := 9;
  end;
  FCurrFont.Name := FFont.Name;
  FCurrFont.Size := FFont.Size;

{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnSrcEditorGutterMgr.Create Gutter Font Size %d', [FFont.Size]);
{$ENDIF}

  FFont.Color := csDefaultLineNumberColor;
  FCurrFont.Color := csDefaultCurrentLineNumberColor;

  FActive := True;
  FShowLineNumber := True;
  FShowLineCount := True;
  FMinWidth := 2;
  FFixedWidth := 4;
  FAutoWidth := True;
  FShowModifier := False;

  FDblClickToggleBookmark := True; // Ĭ������˫���л���ǩ
  FClickSelectLine := False;       // Ĭ�ϲ����õ���ѡ��
  FDragSelectLines := True;        // Ĭ�������϶�ѡ����

  CnWizNotifierServices.AddAfterThemeChangeNotifier(ThemeChanged);
  EditControlWrapper.AddEditControlNotifier(EditControlNotify);
  UpdateGutters;
end;

destructor TCnSrcEditorGutterMgr.Destroy;
var
  I: Integer;
begin
  CnWizNotifierServices.RemoveAfterThemeChangeNotifier(ThemeChanged);
  EditControlWrapper.RemoveEditControlNotifier(EditControlNotify);
  for I := FList.Count - 1 downto 0 do
    TCnSrcEditorGutter(FList[I]).Free;
  FList.Free;
  FFont.Free;
  FCurrFont.Free;
  inherited;
end;

function TCnSrcEditorGutterMgr.CanShowGutter: Boolean;
begin
  Result := FActive and FShowLineNumber;
end;

procedure TCnSrcEditorGutterMgr.DoLineClose;
begin
  ShowLineNumber := False;
end;

procedure TCnSrcEditorGutterMgr.DoEnhConfig;
begin
  if Assigned(FOnEnhConfig) then
    FOnEnhConfig(Self);
end;

procedure TCnSrcEditorGutterMgr.DoUpdateGutters(EditWindow: TCustomForm;
  EditControl: TControl; Context: Pointer);
var
  Gutter: TCnSrcEditorGutter;
begin
  if (EditWindow <> nil) and (EditControl <> nil) then
  begin
    Gutter := TCnSrcEditorGutter(EditWindow.FindComponent(SCnSrcEditorGutter));
    if CanShowGutter then
    begin
      if Gutter = nil then
      begin
        Gutter := TCnSrcEditorGutter.Create(EditWindow);
        Gutter.Name := SCnSrcEditorGutter;
        Gutter.EditControl := EditControl;
        Gutter.EditWindow := EditWindow;
        Gutter.FGutterMgr := Self;
        Gutter.Parent := EditControl.Parent;
        FList.Add(Gutter);
      end;

      Gutter.Font := Font;
      Gutter.FActive := True;
      Gutter.UpdateStatus;
    end
    else if Gutter <> nil then
    begin
      Gutter.FActive := False;
      Gutter.Visible := False;
    end;
  end;
end;

procedure TCnSrcEditorGutterMgr.EditControlNotify(EditControl: TControl;
  EditWindow: TCustomForm; Operation: TOperation);
begin
  UpdateGutters;
end;

procedure TCnSrcEditorGutterMgr.UpdateGutters;
var
  I: Integer;
begin
  EnumEditControl(DoUpdateGutters, nil);
  for I := 0 to Count - 1 do
    Gutters[I].UpdateStatus;
end;

//------------------------------------------------------------------------------
// �������
//------------------------------------------------------------------------------

procedure TCnSrcEditorGutterMgr.LanguageChanged(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    Gutters[I].LanguageChanged(Sender);
end;

procedure TCnSrcEditorGutterMgr.LoadSettings(Ini: TCustomIniFile);
begin
  with TCnIniFile.Create(Ini) do
  try
    FShowLineNumber := ReadBool(csGutter, csShowLineNumber, csShowLineNumberDef);
    FShowLineCount := ReadBool(csGutter, csShowLineCount, True);
{$IFNDEF BDS}
    FShowModifier := ReadBool(csGutter, csShowModifier, csShowModifierDef);
{$ENDIF}
    FFont := ReadFont(csGutter, csFont, FFont);
    FCurrFont := ReadFont(csGutter, csCurrFont, FCurrFont);
    FAutoWidth := ReadBool(csGutter, csAutoWidth, FAutoWidth);
    MinWidth := ReadInteger(csGutter, csMinWidth, FMinWidth);
    FixedWidth := ReadInteger(csGutter, csFixedWidth, FFixedWidth);

    FClickSelectLine := ReadBool(csGutter, csClickSelectLine, FClickSelectLine);
    FDragSelectLines := ReadBool(csGutter, csDragSelectLines, FDragSelectLines);
    FTenMode := ReadBool(csGutter, csTenMode, FTenMode);
    FRelativeNumber := ReadBool(csGutter, csRelativeNumber, FRelativeNumber);
    FDblClickToggleBookmark := ReadBool(csGutter, csDblClickToggleBookmark, FDblClickToggleBookmark);
    UpdateGutters;
  finally
    Free;
  end;
end;

procedure TCnSrcEditorGutterMgr.SaveSettings(Ini: TCustomIniFile);
begin
  with TCnIniFile.Create(Ini) do
  try
    WriteBool(csGutter, csShowLineNumber, FShowLineNumber);
    WriteBool(csGutter, csShowLineCount, FShowLineCount);
{$IFNDEF BDS}
    WriteBool(csGutter, csShowModifier, FShowModifier);
{$ENDIF}
    WriteFont(csGutter, csFont, FFont);
    WriteFont(csGutter, csCurrFont, FCurrFont);
    WriteBool(csGutter, csAutoWidth, FAutoWidth);
    WriteInteger(csGutter, csMinWidth, FMinWidth);
    WriteInteger(csGutter, csFixedWidth, FFixedWidth);
    WriteBool(csGutter, csClickSelectLine, FClickSelectLine);
    WriteBool(csGutter, csDragSelectLines, FDragSelectLines);
    WriteBool(csGutter, csTenMode, FTenMode);
    WriteBool(csGutter, csRelativeNumber, FRelativeNumber);
    WriteBool(csGutter, csDblClickToggleBookmark, FDblClickToggleBookmark);
  finally
    Free;
  end;
end;

procedure TCnSrcEditorGutterMgr.ResetSettings(Ini: TCustomIniFile);
begin

end;

//------------------------------------------------------------------------------
// ���Զ�д
//------------------------------------------------------------------------------

function TCnSrcEditorGutterMgr.GetCount: Integer;
begin
  Result := FList.Count;
end;

function TCnSrcEditorGutterMgr.GetGutters(Index: Integer): TCnSrcEditorGutter;
begin
  Result := TCnSrcEditorGutter(FList[Index]);
end;

procedure TCnSrcEditorGutterMgr.SetFont(const Value: TFont);
begin
  FFont.Assign(Value);
  UpdateGutters;
end;

procedure TCnSrcEditorGutterMgr.SetCurrFont(const Value: TFont);
begin
  FCurrFont.Assign(Value);
  UpdateGutters;
end;

procedure TCnSrcEditorGutterMgr.SetShowLineNumber(const Value: Boolean);
begin
  if Value <> FShowLineNumber then
  begin
    FShowLineNumber := Value;
    UpdateGutters;
  end;
end;

procedure TCnSrcEditorGutterMgr.SetActive(const Value: Boolean);
begin
  if FActive <> Value then
  begin
    FActive := Value;
    UpdateGutters;
  end;
end;

procedure TCnSrcEditorGutterMgr.SetFixedWidth(const Value: TCnGutterWidth);
begin
  FFixedWidth := TrimInt(Value, Low(TCnGutterWidth), High(TCnGutterWidth));
end;

procedure TCnSrcEditorGutterMgr.SetMinWidth(const Value: TCnGutterWidth);
begin
  FMinWidth := TrimInt(Value, Low(TCnGutterWidth), High(TCnGutterWidth));
end;

procedure TCnSrcEditorGutterMgr.SetShowModifier(const Value: Boolean);
begin
  if FShowModifier <> Value then
  begin
    FShowModifier := Value;
    UpdateGutters;
  end;
end;

procedure TCnSrcEditorGutterMgr.SetTenMode(const Value: Boolean);
begin
  if FTenMode <> Value then
  begin
    FTenMode := Value;
    UpdateGutters;
  end;
end;

procedure TCnSrcEditorGutterMgr.ThemeChanged(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
  begin
    try
      if CnThemeWrapper.CurrentIsDark then
        Gutters[I].Color := csDarkBackgroundColor
      else
        Gutters[I].Color := clBtnface;
    except
      ;
    end;
  end;
end;

procedure TCnSrcEditorGutterMgr.SetRelativeNumber(const Value: Boolean);
begin
  if FRelativeNumber <> Value then
  begin
    FRelativeNumber := Value;
    UpdateGutters;
  end;
end;

{ TCnIdentReceiver }

constructor TCnIdentReceiver.Create(AGutter: TCnSrcEditorGutter);
begin
  inherited Create;
  FGutter := AGutter;
  FTimer := TTimer.Create(nil);
  FTimer.Enabled := False;
  FTimer.Interval := csIdentLineShowDelay;
  FTimer.OnTimer := TimerOnTimer;
end;

destructor TCnIdentReceiver.Destroy;
begin
  FTimer.Free;
  inherited;
end;

procedure TCnIdentReceiver.OnEvent(Event: TCnEvent);
begin
  if FGutter <> nil then
  begin
    if Event.EventData = nil then
    begin
      FGutter.FIdentLines.Clear;
      FGutter.FIdentCols.Clear;
    end
    else
    begin
      FGutter.FIdentLines.Assign(TCnList(Event.EventData));
      FGutter.FIdentCols.Assign(TCnList(Event.EventTag));
    end;

{$IFDEF DEBUG}
    CnDebugger.LogFmt('TCnIdentReceiver OnEvent. %d Lines should Paint.', [FGutter.FIdentLines.Count]);
{$ENDIF}

    FTimer.Enabled := False;
    FTimer.Enabled := True;
  end;
end;

procedure TCnIdentReceiver.TimerOnTimer(Sender: TObject);
begin
  if FGutter <> nil then
    FGutter.Invalidate;
end;

{$ENDIF CNWIZARDS_CNSRCEDITORENHANCE}
end.
