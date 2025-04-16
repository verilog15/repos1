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

unit CnEditorExtSelect;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��㼶����ѡ��ʵ�ֵ�Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin7 SP2 + Delphi 5.01
* ���ݲ��ԣ�PWin7 + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2021.10.06 V1.0
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls, IniFiles, Menus, ToolsAPI,
  CnWizUtils, CnConsts, CnCommon, CnWizManager, CnWizEditFiler,
  CnCodingToolsetWizard, CnWizConsts, CnSelectionCodeTool, CnWizIdeUtils,
  CnSourceHighlight, CnPasCodeParser, CnEditControlWrapper, mPasLex,
  CnCppCodeParser, mwBCBTokenList;

type
  TCnEditorExtendingSelect = class(TCnBaseCodingToolset)
  private
    FLevel: Integer;
    FTimer: TTimer;
    FNeedReparse: Boolean;
    FWholeLines: Boolean;
    FSelecting: Boolean;
    FStartPos, FEndPos: TOTACharPos;
    procedure CheckModifiedAndReparse;
    procedure EditorChanged(Editor: TEditorObject; ChangeType:
      TEditorChangeTypes);
    procedure OnSelectTimer(Sender: TObject);
  protected
    function GetDefShortCut: TShortCut; override;
  public
    constructor Create(AOwner: TCnCodingToolsetWizard); override;
    destructor Destroy; override;

    function GetCaption: string; override;
    function GetHint: string; override;
    procedure GetToolsetInfo(var Name, Author, Email: string); override;
    procedure Execute; override;

    property WholeLines: Boolean read FWholeLines write FWholeLines;
    {* ��ѡ��ʱ�Ƿ�������ģʽ}
  end;

implementation

uses
  CnDebug;

{ TCnEditorExtendingSelect }

procedure TCnEditorExtendingSelect.CheckModifiedAndReparse;
const
  NO_LAYER = -2;
var
  EditView: IOTAEditView;
  EditControl: TControl;
  CurrIndex, I: Integer;
  PasParser: TCnGeneralPasStructParser;
  CppParser: TCnGeneralCppStructParser;
  Stream: TMemoryStream;
  EditPos: TOTAEditPos;
  CharPos: TOTACharPos;
  CurrentToken: TCnGeneralPasToken;
  CurrentTokenName: TCnIdeTokenString;
  CurIsPas, CurIsCpp: Boolean;
  CurrentTokenIndex: Integer;
  BlockMatchInfo: TCnBlockMatchInfo;
  MaxInnerLayer, MinOutLayer: Integer;
  Pair: TCnBlockLinePair;
  LastS: string;

  // �ж�һ�� Pair �Ƿ�����˹��λ��
  function EditPosInPair(AEditPos: TOTAEditPos; APair: TCnBlockLinePair): Boolean;
  var
    AfterStart, BeforeEnd: Boolean;
  begin
    AfterStart := (AEditPos.Line > APair.StartToken.EditLine) or
      ((AEditPos.Line = APair.StartToken.EditLine) and (AEditPos.Col >= APair.StartToken.EditCol));
    BeforeEnd := (AEditPos.Line < APair.EndToken.EditLine) or
      ((AEditPos.Line = APair.EndToken.EditLine) and
      (AEditPos.Col <= APair.EndToken.EditEndCol));

    Result := AfterStart and BeforeEnd;
  end;

begin
  EditControl := CnOtaGetCurrentEditControl;
  if EditControl = nil then
    Exit;
  try
    EditView := EditControlWrapper.GetEditView(EditControl);
  except
    Exit;
  end;

  if EditView = nil then
    Exit;

  CurIsPas := IsDprOrPas(EditView.Buffer.FileName) or IsInc(EditView.Buffer.FileName);
  CurIsCpp := IsCppSourceModule(EditView.Buffer.FileName);
  if (not CurIsCpp) and (not CurIsPas) then
    Exit;

  // ����
  PasParser := nil;
  CppParser := nil;
  BlockMatchInfo := nil;

  try
    if CurIsPas then
    begin
      PasParser := TCnGeneralPasStructParser.Create;
  {$IFDEF BDS}
      PasParser.UseTabKey := True;
      PasParser.TabWidth := EditControlWrapper.GetTabWidth;
  {$ENDIF}
    end;

    if CurIsCpp then
    begin
      CppParser := TCnGeneralCppStructParser.Create;
  {$IFDEF BDS}
      CppParser.UseTabKey := True;
      CppParser.TabWidth := EditControlWrapper.GetTabWidth;
  {$ENDIF}
    end;

    Stream := TMemoryStream.Create;
    try
      CnGeneralSaveEditorToStream(EditView.Buffer, Stream);

      // ������ǰ��ʾ��Դ�ļ�
      if CurIsPas then
        CnPasParserParseSource(PasParser, Stream, IsDpr(EditView.Buffer.FileName)
          or IsInc(EditView.Buffer.FileName), False);
      if CurIsCpp then
        CnCppParserParseSource(CppParser, Stream, EditView.CursorPos.Line, EditView.CursorPos.Col);
    finally
      Stream.Free;
    end;

    if CurIsPas then
    begin
      // �������ٲ��ҵ�ǰ������ڵĿ飬��ֱ��ʹ�� CursorPos����Ϊ Parser ����ƫ�ƿ��ܲ�ͬ
      CnOtaGetCurrentCharPosFromCursorPosForParser(CharPos);
      PasParser.FindCurrentBlock(CharPos.Line, CharPos.CharIndex);
    end;

    BlockMatchInfo := TCnBlockMatchInfo.Create(EditControl);
    BlockMatchInfo.LineInfo := TCnBlockLineInfo.Create(EditControl);

    // �����õ� Token ���� BlockMatchInfo ��
    for I := 0 to PasParser.Count - 1 do
      if PasParser.Tokens[I].TokenID in csKeyTokens + [tkSemiColon] then
        BlockMatchInfo.AddToKeyList(PasParser.Tokens[I]);

    // ת��һ��
    for I := 0 to BlockMatchInfo.KeyCount - 1 do
      ConvertGeneralTokenPos(Pointer(EditView), BlockMatchInfo.KeyTokens[I]);

    // ������
    BlockMatchInfo.IsCppSource := CurIsCpp;
    BlockMatchInfo.CheckLineMatch(EditView, False, False);

    // BlockMatchInfo ������� LineInfo �ڵ�����

    FStartPos.Line := -1;
    FEndPos.Line := -1;
    MaxInnerLayer := NO_LAYER; // -2 ������
    MinOutLayer := MaxInt;
    EditPos := EditView.CursorPos;

    // �õ�������� Pair �������
    for I := 0 to BlockMatchInfo.LineInfo.Count - 1 do
    begin
      // ���ҿ���λ�õ����ڲ�Ҳ���� Layer ��ߵ� Pair
      Pair := BlockMatchInfo.LineInfo.Pairs[I];
      if EditPosInPair(EditPos, Pair) then
      begin
        if Pair.Layer > MaxInnerLayer then
          MaxInnerLayer := Pair.Layer;
        if Pair.Layer < MinOutLayer then
          MinOutLayer := Pair.Layer;
      end;
    end;

{$IFDEF DEBUG}
    CnDebugger.LogFmt('CheckModifiedAndReparse Get Layer from %d to %d.', [MinOutLayer, MaxInnerLayer]);
{$ENDIF}

    if (MaxInnerLayer = NO_LAYER) or (MinOutLayer = MaxInt) then
      Exit;

    // Layer �� MinOutLayer������ -1 �� 0�� �� MaxInnerLayer ��FLevel �� 1 ���⣬FLevel �� Layer �и����Զ�Ӧ��ϵ
    // FLevel 1 <=> MaxInnerLayer��FLevel 2 <=> MaxInnerLayer - 1��... MaxLevel <=> MinOutLayer
    // ���� FLevel + Layer = 1 + MaxInnerLayer ���� MaxLevel := MaxInnerLayer + 1 - MinOutLayer
    if FLevel > MaxInnerLayer + 1 - MinOutLayer then
    begin
      // ȫѡ�����ļ�
      FStartPos.Line := 1;
      FStartPos.CharIndex := 0;
      FEndPos.Line := EditView.Buffer.GetLinesInBuffer;
      LastS := CnOtaGetLineText(FEndPos.Line, EditView.Buffer);
      FEndPos.CharIndex := Length(LastS);
      Exit;
    end;

    for I := 0 to BlockMatchInfo.LineInfo.Count - 1 do
    begin
      // ���ҿ���λ�õ����ڲ�Ҳ���� Layer ��ߵ� Pair
      Pair := BlockMatchInfo.LineInfo.Pairs[I];
      if Pair.Layer = MaxInnerLayer + 1 - FLevel then
      begin
        if EditPosInPair(EditPos, Pair) then
        begin
          FStartPos.Line := Pair.StartToken.EditLine;
          FStartPos.CharIndex := Pair.StartToken.EditCol;
          FEndPos.Line := Pair.EndToken.EditLine;
          FEndPos.CharIndex := Pair.EndToken.EditEndCol;
        end;
      end;
    end;
  finally
    BlockMatchInfo.LineInfo.Free;
    BlockMatchInfo.LineInfo := nil;
    BlockMatchInfo.Free; // LineInfo �� nil ������� Clear ���ܽ���
    PasParser.Free;
    CppParser.Free;
  end;
end;

constructor TCnEditorExtendingSelect.Create(AOwner: TCnCodingToolsetWizard);
begin
  inherited;
  FTimer := TTimer.Create(nil);
  FTimer.Interval := 500;
  FTimer.OnTimer := OnSelectTimer;
  EditControlWrapper.AddEditorChangeNotifier(EditorChanged);
end;

destructor TCnEditorExtendingSelect.Destroy;
begin
  EditControlWrapper.RemoveEditorChangeNotifier(EditorChanged);
  FTimer.Free;
  inherited;
end;

procedure TCnEditorExtendingSelect.EditorChanged(Editor: TEditorObject;
  ChangeType: TEditorChangeTypes);
begin
  if ChangeType * [ctView, ctModified, ctTopEditorChanged, ctOptionChanged] <> [] then
    FNeedReparse := True;
  if not FSelecting and (ChangeType * [ctBlock] <> []) then
    FLevel := 0;
end;

procedure TCnEditorExtendingSelect.Execute;
var
  CurrIndex: Integer;
  EditView: IOTAEditView;
  CurrTokenStr: TCnIdeTokenString;
begin
  EditView := CnOtaGetTopMostEditView;
  if EditView = nil then
    Exit;

  // ���������
  // ���û��ѡ��������ѡ��ǰ��ʶ������ Level 1���ޱ�ʶ���Ļ�������ѡ���ڲ㿪���䣬���� Level 2
  // �����ѡ������������������ݵ�ǰ Level ������ 1 ѡ��
  // �㼶����˳����ѡ���� 0������±�ʶ�� 1����һ���У�����ǰ����ڿ����� 2��
  // ��ǰ������䣨Ҳ���������飬�����зֺžͼӸ��ֺţ�3
  // �͵�ǰ��ͬ�������п飨�������������Ļ���4��������ڿ����� 5���Դ�����

  FSelecting := True;
  try
    if (EditView.Block = nil) or not EditView.Block.IsValid then
    begin
      if CnOtaGeneralGetCurrPosToken(CurrTokenStr, CurrIndex) then
      begin
        if CurrTokenStr <> '' then
        begin
          // ������б�ʶ����ѡ��
          CnOtaSelectCurrentToken;
          Exit;
        end;
      end;
    end;

    Inc(FLevel);
{$IFDEF DEBUG}
    CnDebugger.LogFmt('EditorExtendingSelect To Select Level %d.', [FLevel]);
{$ENDIF}

    CheckModifiedAndReparse;

    // ѡ�� FLevel ��Ӧ����
    if (FStartPos.Line >= 0) and (FEndPos.Line >= 0) then
      CnOtaMoveAndSelectBlock(FStartPos, FEndPos);
  finally
    FTimer.Enabled := False;
    FTimer.Enabled := True; // �����Ӻ����� FSelecting
  end;
end;

function TCnEditorExtendingSelect.GetCaption: string;
begin
  Result := SCnEditorExtendingSelectMenuCaption;
end;

function TCnEditorExtendingSelect.GetDefShortCut: TShortCut;
begin
  Result := TextToShortCut('Alt+Q');
end;

procedure TCnEditorExtendingSelect.GetToolsetInfo(var Name, Author,
  Email: string);
begin
  Name := SCnEditorExtendingSelectName;
  Author := SCnPack_LiuXiao;
  Email := SCnPack_LiuXiaoEmail;
end;

function TCnEditorExtendingSelect.GetHint: string;
begin
  Result := SCnEditorExtendingSelectMenuHint;
end;

procedure TCnEditorExtendingSelect.OnSelectTimer(Sender: TObject);
begin
  FSelecting := False;
end;

initialization
  RegisterCnCodingToolset(TCnEditorExtendingSelect);

end.
