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

unit CnProjectListUsedFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ������õ�Ԫ�б���
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWinXPPro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2018.03.29 V1.1
*               �ع���֧��ģ��ƥ��
*           2007.07.03 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNPROJECTEXTWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Controls, Forms, Dialogs, Contnrs,
{$IFDEF COMPILER6_UP}
  StrUtils,
{$ENDIF}
  ComCtrls, StdCtrls, ExtCtrls, Math, ToolWin, Clipbrd, IniFiles, ToolsAPI,
  Graphics, ImgList, ActnList,
  CnPasCodeParser,  CnWizIdeUtils, CnEditorOpenFile, CnWizUtils, CnIni,
  CnCommon, CnConsts, CnWizConsts, CnWizOptions, CnWizMultiLang,
  CnProjectViewBaseFrm, CnProjectViewUnitsFrm, CnLangMgr, CnStrings;

type

//==============================================================================
// �����õ�Ԫ�б���
//==============================================================================

{ TCnProjectListUsedForm }

  TCnProjectListUsedForm = class(TCnProjectViewBaseForm)
    procedure lvListData(Sender: TObject; Item: TListItem);
  private
    FCurFile: string;
    FIsDpr: Boolean;
    FIsPas: Boolean;
    FIsC: Boolean;
  protected
    function DoSelectOpenedItem: string; override;
    procedure OpenSelect; override;
    function GetSelectedFileName: string; override;
    function GetHelpTopic: string; override;
    procedure CreateList; override;
    procedure UpdateComboBox; override;
    procedure UpdateStatusBar; override;
    procedure DoLanguageChanged(Sender: TObject); override;

    function SortItemCompare(ASortIndex: Integer; const AMatchStr: string;
      const S1, S2: string; Obj1, Obj2: TObject; SortDown: Boolean): Integer; override;
    function CanMatchDataByIndex(const AMatchStr: string; AMatchMode: TCnMatchMode;
      DataListIndex: Integer; var StartOffset: Integer; MatchedIndexes: TList): Boolean; override;
    procedure DrawListPreParam(Item: TListItem; ListCanvas: TCanvas); override;
  public
    class procedure ParseUnitInclude(const Source: string; UsesList: TStrings);
  end;

function ShowProjectListUsed(Ini: TCustomIniFile): Boolean;

{$ENDIF CNWIZARDS_CNPROJECTEXTWIZARD}

implementation

{$IFDEF CNWIZARDS_CNPROJECTEXTWIZARD}

{$R *.DFM}

{$IFDEF DEBUG}
uses
  CnDebug;
{$ENDIF}

const
  csListUsed = 'ListUsed';

type
  TControlAccess = class(TControl);

  TCnUsedUnitInfo = class(TCnBaseElementInfo)
  private
    FInImpl: Boolean;
  public
    property InImpl: Boolean read FInImpl write FInImpl;
  end;

function ShowProjectListUsed(Ini: TCustomIniFile): Boolean;
begin
  with TCnProjectListUsedForm.Create(nil) do
  begin
    try
      ShowHint := WizOptions.ShowHint;
      LoadSettings(Ini, csListUsed);
      Result := ShowModal = mrOk;
      SaveSettings(Ini, csListUsed);
      if Result then
        BringIdeEditorFormToFront;
    finally
      Free;
    end;
  end;
end;

//==============================================================================
// �����õ�Ԫ�б���
//==============================================================================

{ TCnProjectListUsedForm }

procedure TCnProjectListUsedForm.CreateList;
var
  Stream: TMemoryStream;
  TmpName: string;
  I: Integer;
  Info: TCnUsedUnitInfo;
begin
  FCurFile := CnOtaGetCurrentSourceFile;

  if FCurFile <> '' then
  begin
    if IsForm(FCurFile) then
    begin
      TmpName := _CnChangeFileExt(FCurFile, '.pas');
      if CnOtaIsFileOpen(TmpName) then
        FCurFile := TmpName
      else
      begin
        TmpName := _CnChangeFileExt(FCurFile, '.cpp');
        if CnOtaIsFileOpen(TmpName) then
          FCurFile := TmpName;
      end;
    end;

    Caption := Caption + ' - ' + FCurFile;
    Stream := TMemoryStream.Create;
    try
      CnOtaSaveCurrentEditorToStream(Stream, False);
      if IsDelphiSourceModule(FCurFile) then
      begin
        FIsPas := True;
        FIsDpr := IsDpr(FCurFile);
        ParseUnitUses(PAnsiChar(Stream.Memory), DataList);

        // ParseUnitUses ���������Ƿ��� Implementation ���ֵ��Ǹ� Boolean��ת�ɶ���
        for I := 0 to DataList.Count - 1 do
        begin
          Info := TCnUsedUnitInfo.Create;
          Info.Text := DataList[I];

          if DataList.Objects[I] <> nil then
            Info.InImpl := True;

          DataList.Objects[I] := Info;
        end;
      end
      else if IsCppSourceModule(FCurFile) then
      begin
        // ���� C �� include
        FIsC := True;
        ParseUnitInclude(PChar(Stream.Memory), DataList);
        // ͬ��ת�ɶ���
        for I := 0 to DataList.Count - 1 do
        begin
          Info := TCnUsedUnitInfo.Create;
          Info.Text := DataList[I];
          DataList.Objects[I] := Info;
        end;
      end;
    finally
      Stream.Free;
    end;

{$IFDEF DEBUG}
    CnDebugger.LogStrings(DataList, 'Used List.');
{$ENDIF}
  end;
end;

function TCnProjectListUsedForm.GetHelpTopic: string;
begin
  Result := 'CnProjectExtListUsed';
end;

procedure TCnProjectListUsedForm.OpenSelect;
var
  I: Integer;
  Error: Boolean;
{$IFDEF SUPPORT_UNITNAME_DOT}
  Prefix: string;
  Prefixes: TStrings;
  PO: IOTAProjectOptions;
{$ENDIF}
begin
  Error := False;
  if lvList.SelCount > 0 then
  begin
    if (lvList.SelCount > 1) and actQuery.Checked then
      if not QueryDlg(SCnProjExtOpenUnitWarning, False, SCnInformation) then
        Exit;

{$IFDEF SUPPORT_UNITNAME_DOT}
    Prefixes := TStringList.Create;
    try
      PO := CnOtaGetActiveProjectOptions;
      if PO <> nil then
      begin
        Prefix := PO.Values['NamespacePrefix'];
        if Trim(Prefix) <> '' then
          ExtractStrings([';'], [' '], PChar(Prefix), Prefixes);
      end;
{$ENDIF}

      for I := 0 to lvList.Items.Count - 1 do
      begin
        if lvList.Items[I].Selected then
        begin
          if not TCnEditorOpenFile.SearchAndOpenFile(lvList.Items[I].Caption
            {$IFDEF SUPPORT_UNITNAME_DOT}, Prefixes {$ENDIF}) then
          begin
            Error := True;
            ErrorDlg(SCnEditorOpenFileNotFound);
          end;
        end;
      end;

{$IFDEF SUPPORT_UNITNAME_DOT}
    finally
      Prefixes.Free;
    end;
{$ENDIF}

    if not Error then
      ModalResult := mrOK;    
  end;
end;

procedure TCnProjectListUsedForm.UpdateStatusBar;
begin
  StatusBar.Panels[1].Text := Format(SCnProjExtUnitsFileCount, [DisplayList.Count]);
end;

procedure TCnProjectListUsedForm.lvListData(Sender: TObject;
  Item: TListItem);
var
  Info: TCnUsedUnitInfo;
begin
  if (DisplayList <> nil) and (Item.Index >= 0) and
    (Item.Index < DisplayList.Count) then
  begin
    Item.Caption := DisplayList[Item.Index];
    Item.ImageIndex := 78; // Unit
    if FIsDpr then
      Item.SubItems.Add('project')
    else if FIsC then
      Item.SubItems.Add('include')
    else
    begin
      Info := TCnUsedUnitInfo(DisplayList.Objects[Item.Index]);
      if (Info = nil) or not Info.InImpl then
        Item.SubItems.Add('interface')
      else
        Item.SubItems.Add('implementation');
    end;
    Item.Data := DisplayList.Objects[Item.Index];
    RemoveListViewSubImages(Item);
  end;
end;

procedure TCnProjectListUsedForm.UpdateComboBox;
begin
// Do nothing for Combo Hidden.
end;

function TCnProjectListUsedForm.DoSelectOpenedItem: string;
var
  CurrentModule: IOTAModule;
begin
  CurrentModule := CnOtaGetCurrentModule;
  Result := _CnChangeFileExt(_CnExtractFileName(CurrentModule.FileName), '');
end;

function TCnProjectListUsedForm.GetSelectedFileName: string;
begin
  if Assigned(lvList.ItemFocused) then
    Result := Trim(lvList.ItemFocused.Caption);
end;

class procedure TCnProjectListUsedForm.ParseUnitInclude(
  const Source: string; UsesList: TStrings);
const
  SCnInclude = '#include';
var
  I, J, QS, QE, BS, BE, Len: Integer;
begin
  Len := Length(SCnInclude);
  if (UsesList <> nil) and (Source <> '') then
  begin
    UsesList.Text := Source;
    for I := UsesList.Count - 1 downto 0 do
    begin
      if AnsiStartsText(SCnInclude, Trim(UsesList[I])) then
      begin
        UsesList[I] := Trim(Copy(Trim(UsesList[I]), Len + 1, MaxInt));
        QS := 0; QE := 0; BS := 0; BE := 0;
        for J := 1 to Length(UsesList[I]) do
        begin
          case UsesList[I][J] of
          '"':
            begin
              if QS = 0 then
                QS := J
              else
                QE := J;
            end;
          '<':
            BS := J;
          '>':
            BE := J;
          end;
        end;

        if (BE > 0) and (BS > 0) and (BE > BS) then
          UsesList[I] := Copy(UsesList[I], BS + 1, BE - BS - 1)
        else if (QE > 0) and (QS > 0) and (QE > QS) then
          UsesList[I] := Copy(UsesList[I], QS + 1, QE - QS - 1);

        if Length(UsesList[I]) = 0 then
          UsesList.Delete(I);
      end
      else
        UsesList.Delete(I);
    end;
  end;
end;

procedure TCnProjectListUsedForm.DoLanguageChanged(Sender: TObject);
begin
  try
    ToolBar.ShowCaptions := True;
    ToolBar.ShowCaptions := False;
  except
    ;
  end;
end;

function TCnProjectListUsedForm.CanMatchDataByIndex(
  const AMatchStr: string; AMatchMode: TCnMatchMode;
  DataListIndex: Integer; var StartOffset: Integer; MatchedIndexes: TList): Boolean;
begin
  Result := False;
  if AMatchStr = '' then
  begin
    Result := True;
    Exit;
  end;

  case AMatchMode of // ����ʱֻ�����Ʋ���ƥ�䣬�����ִ�Сд
    mmStart:
      begin
        Result := Pos(UpperCase(AMatchStr), UpperCase(DataList[DataListIndex])) = 1;
      end;
    mmAnywhere:
      begin
        Result := Pos(UpperCase(AMatchStr), UpperCase(DataList[DataListIndex])) > 0;
      end;
    mmFuzzy:
      begin
        Result := FuzzyMatchStr(AMatchStr, DataList[DataListIndex], MatchedIndexes);
      end;
  end;
end;

function TCnProjectListUsedForm.SortItemCompare(ASortIndex: Integer;
  const AMatchStr, S1, S2: string; Obj1, Obj2: TObject; SortDown: Boolean): Integer;
begin
  case ASortIndex of // ��Ϊ����ʱ���Ʋ���ƥ�䣬�������ʱҲҪ���ǵ������Ƶ�ȫƥ����ǰ
    0:
      begin
        Result := CompareTextWithPos(AMatchStr, S1, S2, SortDown);
      end;
    1:
      begin
        Result := Integer(Obj1) - Integer(Obj2);
        if SortDown then
          Result := -Result;
      end;
  else
    Result := 0;
  end;
end;

procedure TCnProjectListUsedForm.DrawListPreParam(Item: TListItem;
  ListCanvas: TCanvas);
begin

end;

{$ENDIF CNWIZARDS_CNPROJECTEXTWIZARD}
end.
