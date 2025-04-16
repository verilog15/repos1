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

unit CnWizUpgradeFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ������������Զ���ⵥԪ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע�������������Զ���ⵥԪ
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2003.08.10 V1.1
*               ֻ�е�����رա���ťʱ���Ŵ����Ժ�����ʾ������
*           2003.04.28 V1.1
*               �������� IDE �����Ϲرտ��ܵ��� IDE ����������
*           2003.03.09 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  WinInet, IniFiles, CnWizConsts, CnWizOptions, CnCommon, StdCtrls, ExtCtrls,
  CnWizUtils, CnInetUtils, CnWizMultiLang, CnWizCompilerConst, CnWideCtrls,
  CnWideStrings;

type

{ TCnWizUpgradeItem }

  TCnWizUpgradeItem = class(TCollectionItem)
  private
    FBigBugFixed: Boolean;
    FNewFeature: Boolean;
    FVersion: string;
    FDate: TDateTime;
    FComment: string;
    FURL: string;
    FBetaVersion: Boolean;
    FWideComment: WideString;
  public
    procedure Assign(Source: TPersistent); override;
  published
    property Date: TDateTime read FDate write FDate;
    property Version: string read FVersion write FVersion;
    property NewFeature: Boolean read FNewFeature write FNewFeature;
    property BigBugFixed: Boolean read FBigBugFixed write FBigBugFixed;
    property BetaVersion: Boolean read FBetaVersion write FBetaVersion;
    property Comment: string read FComment write FComment;
    property WideComment: WideString read FWideComment write FWideComment;
    property URL: string read FURL write FURL;
  end;

{ TCnWizUpgradeCollection }

  TCnWizUpgradeCollection = class(TCollection)
  private
    function GetItems(Index: Integer): TCnWizUpgradeItem;
    procedure SetItems(Index: Integer; const Value: TCnWizUpgradeItem);
  public
    constructor Create;
    function Add: TCnWizUpgradeItem;
    property Items[Index: Integer]: TCnWizUpgradeItem read GetItems write SetItems; default;
  end;

{ TCnWizUpgradeThread }

  TCnWizUpgradeThread = class(TThread)
  private
    FUserCheck: Boolean;
    FUpgradeCollection: TCnWizUpgradeCollection;
    FHTTP: TCnHTTP;
    function GetUpgradeCollection(const Content: string
      {$IFNDEF UNICODE}; WideCon: WideString {$ENDIF}): Boolean;
    procedure CheckUpgrade;
    procedure FindLinks(S: string; Strings: TStrings);
    function GetUpgrade(const AURL: string; Level: Integer): Boolean;
  protected

  public
    procedure Execute; override;
    constructor Create(AUserCheck: Boolean);
    destructor Destroy; override;
  end;

{ TCnWizUpgradeForm }

  TCnWizUpgradeForm = class(TCnTranslateForm)
    pnlTop: TPanel;
    Label1: TLabel;
    Bevel2: TBevel;
    pnlBottom: TPanel;
    cbNoHint: TCheckBox;
    btnDownload: TButton;
    Bevel1: TBevel;
    btnClose: TButton;
    btnHelp: TButton;
    pnlLeft: TPanel;
    Image1: TImage;
    pnlRight: TPanel;
    lbl1: TLabel;
    procedure btnDownloadClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure btnHelpClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure btnCloseClick(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure cbNoHintClick(Sender: TObject);
  private
    FCollection: TCnWizUpgradeCollection;
    FMemo: TMemo;
    FPanel: TPanel;
    FLabelContent: TCnWideLabel;
    function NeedManuallyUnicode: Boolean;
  protected
    function GetHelpTopic: string; override;
  public

  end;

procedure CheckUpgrade(AUserCheck: Boolean);

implementation

uses
{$IFDEF DEBUG}
  CnDebug,
{$ENDIF}
  CnWizTipOfDayFrm;

{$R *.DFM}

const
  csVersion = 'Version';
  csNewFeature = 'NewFeature';
  csBigBugFixed = 'BigBugFixed';
  csBetaVersion = 'BetaVersion';
  csURL = 'URL';
  csURLCN = 'URL_CN';
  csNoHint = 'NoHint';

var
  FThread: TCnWizUpgradeThread;
  FForm: TCnWizUpgradeForm;

procedure CheckUpgrade(AUserCheck: Boolean);
begin
  // ��ʱ����Ĵ��룬�ĳ�һ��ֻ���һ�Σ��¿���в���ʹ�ñ���Ԫ
  with WizOptions do
  begin
    // ��ֹ�û���ǰ��������
    if Date < UpgradeCheckDate then
      UpgradeCheckDate := Date - 1;
    if AUserCheck or ((UpgradeStyle = usAllUpgrade) or (UpgradeStyle = usUserDefine) and
      (UpgradeContent <> [])) and (Date - UpgradeCheckDate >= 1) then
    begin
      UpgradeCheckDate := Date;
      if FThread = nil then
      begin
        FThread := TCnWizUpgradeThread.Create(AUserCheck);
      end
      else
        FThread.FUserCheck := AUserCheck;
    end;
  end;
end;

{ TCnWizUpgradeItem }

procedure TCnWizUpgradeItem.Assign(Source: TPersistent);
begin
  if Source is TCnWizUpgradeItem then
  begin
    FBigBugFixed := TCnWizUpgradeItem(Source).FBigBugFixed;
    FNewFeature := TCnWizUpgradeItem(Source).FNewFeature;
    FVersion := TCnWizUpgradeItem(Source).FVersion;
    FDate := TCnWizUpgradeItem(Source).FDate;
    FComment := TCnWizUpgradeItem(Source).FComment;
    FURL := TCnWizUpgradeItem(Source).FURL;
    FBetaVersion := TCnWizUpgradeItem(Source).FBetaVersion;
    FWideComment := TCnWizUpgradeItem(Source).FWideComment;
  end
  else
    inherited;
end;

{ TCnWizUpgradeCollection }

constructor TCnWizUpgradeCollection.Create;
begin
  inherited Create(TCnWizUpgradeItem);
end;

function TCnWizUpgradeCollection.Add: TCnWizUpgradeItem;
begin
  Result := TCnWizUpgradeItem(inherited Add);
end;

function TCnWizUpgradeCollection.GetItems(Index: Integer): TCnWizUpgradeItem;
begin
  Result := TCnWizUpgradeItem(inherited Items[Index]);
end;

procedure TCnWizUpgradeCollection.SetItems(Index: Integer;
  const Value: TCnWizUpgradeItem);
begin
  inherited Items[Index] := Value;
end;

{ TCnWizUpgradeThread }

constructor TCnWizUpgradeThread.Create(AUserCheck: Boolean);
begin
  inherited Create(True);
  FUserCheck := AUserCheck;
  FreeOnTerminate := True;
  FUpgradeCollection := TCnWizUpgradeCollection.Create;
  FHTTP := TCnHTTP.Create;
  if FForm <> nil then FForm.Close;
  
  if Suspended then
    Resume;
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnWizUpgradeThread.Create');
{$ENDIF}
end;

destructor TCnWizUpgradeThread.Destroy;
begin
  FThread := nil;
  FHTTP.Free;
  FUpgradeCollection.Free;
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnWizUpgradeThread.Destroy');
{$ENDIF}
  inherited;
end;

const
  csHttp = 'http://';
  csMaxLevel = 3;
  csMaxLinks = 3;

procedure TCnWizUpgradeThread.Execute;
var
  S: string;
  Y1, M1, D1, Y2, M2, D2: Word;
begin
  // ȡ������¼������ IDE ��汾����ר�Ұ��汾���Լ����� ID ��Ϊ����
  S := Format('%s?ide=%s&ver=%s&langid=%d', [WizOptions.UpgradeURL, CompilerShortName,
    SCnWizardVersion, WizOptions.CurrentLangID]);

  // �ֶ�����
  if FUserCheck then
    S := S + '&manual=1';

  // ÿ���µ���һ��
  DecodeDate(WizOptions.UpgradeCheckMonth, Y1, M1, D1);
  DecodeDate(Date, Y2, M2, D2);
  if (Y1 <> Y2) or (M1 <> M2) then
  begin
    S := S + '&month=1';
    WizOptions.UpgradeCheckMonth := Date;
  end;

  // ȡ�ø�����Ϣ
  if not GetUpgrade(S, 1) then
  begin
    if not FHTTP.Aborted and FUserCheck then
      ErrorDlg(SCnWizUpgradeFail);
  end;

  // ����°汾
  if not FHTTP.Aborted and not Terminated then
    Synchronize(CheckUpgrade);
end;

// �� S ���ҳ����õ� URL ����
procedure TCnWizUpgradeThread.FindLinks(S: string; Strings: TStrings);
var
  I, J: Integer;
begin
  Strings.Clear;
  I := Pos(csHttp, LowerCase(S));
  while I > 0 do
  begin
    J := I + Length(csHttp);
    while (J < Length(S)) and not CharInSet(S[J], ['"', ' ', '>']) do
      Inc(J);
    Strings.Add(Copy(S, I, J - I));
    Delete(S, I, J - I);
    I := Pos(csHttp, LowerCase(S));
  end;
end;

function TCnWizUpgradeThread.GetUpgrade(const AURL: string; Level: Integer): Boolean;
var
  Content: string;
{$IFNDEF UNICODE}
  WideCon: WideString;
{$ENDIF}
  Res: AnsiString;
  Strings: TStrings;
  I: Integer;
begin
  Result := False;

  // �µ������ļ�����ݶ��� UTF8 ����
  Res := TrimBom(FHTTP.GetString(AURL));
{$IFDEF UNICODE}
  Content := UTF8ToString(Res);
{$ELSE}
  Content := CnUtf8ToAnsi(Res);
  WideCon := CnUtf8DecodeToWideString(Res);
{$ENDIF}

{$IFDEF DEBUG}
  CnDebugger.LogMsg('Upgrade: ' + AURL);
  CnDebugger.LogMsg(Content);
{$ENDIF}
  if FHTTP.GetDataFail or FHTTP.Aborted then
    Exit;

  // �ӷ��ؽ��ȡ��������
  if GetUpgradeCollection(Content{$IFNDEF UNICODE}, WideCon {$ENDIF}) then
  begin
    Result := True;
    Exit;
  end
  else if Level <= csMaxLevel then    // ��ת�����ݣ��ٷ�����ת���ַ
  begin                               // ת��ݹ鲻�ܳ���ָ����
    Strings := TStringList.Create;
    try
      FindLinks(Content, Strings);
      if Strings.Count <= csMaxLinks then // ������ת����Ϣ��Ӧ���й��������
        for I := 0 to Strings.Count - 1 do
          if GetUpgrade(Strings[I], Level + 1) then
          begin
            Result := True;
            Exit;
          end
          else if FHTTP.Aborted or Terminated then
            Exit;
    finally
      Strings.Free;
    end;
  end;
end;

function TCnWizUpgradeThread.GetUpgradeCollection(const Content: string
  {$IFNDEF UNICODE}; WideCon: WideString {$ENDIF}): Boolean;
var
  Strings: TStrings;
  Ini: TMemIniFile;
  I: Integer;
{$IFNDEF UNICODE}
  ADateStr: WideString;
  Idx: Integer;
{$ENDIF}
  ADate: TDateTime;
  Item: TCnWizUpgradeItem;
begin
  FUpgradeCollection.Clear;
  Strings := nil;
  Ini := nil;
  try
    Strings := TStringList.Create;
    Ini := TMemIniFile.Create('');
    Strings.Text := Content;
    if Strings.Count > 0 then
    begin
      Ini.SetStrings(Strings);
      Ini.ReadSections(Strings);
      for I := 0 to Strings.Count - 1 do
      begin
        try
{$IFNDEF UNICODE}
          ADateStr := WideString(Strings[I]);
{$ENDIF}
          ADate := CnStrToDate(Strings[I]);
          Item := FUpgradeCollection.Add;
          with Item do
          begin
            Date := ADate;
            Version := Ini.ReadString(Strings[I], csVersion, '');
            NewFeature := Ini.ReadBool(Strings[I], csNewFeature, False);
            BigBugFixed := Ini.ReadBool(Strings[I], csBigBugFixed, False);
            Comment := StrToLines(Ini.ReadString(Strings[I], SCnWizUpgradeCommentName, ''));
{$IFNDEF UNICODE}
            Idx := Pos(ADateStr, WideCon);
            if Idx > 0 then
            begin
              Delete(WideCon, 1, Idx - 1);
              Idx := Pos(SCnWizUpgradeCommentName + '=', WideCon);
              if Idx > 0 then
              begin
                Delete(WideCon, 1, Idx - 1);
                Idx := Pos(#13#10, WideCon);
                if Idx > 0 then
                begin
                  // From 1 to Idx - 1 is Comment
                  WideComment := Copy(WideCon, Length(SCnWizUpgradeCommentName) + 2,
                    Idx - 1 - (Length(SCnWizUpgradeCommentName) + 1));
                  WideComment := WideStrToLines(WideComment);
                end;
              end;
            end;
            // Find ADate in WideComment and delete before, Find SCnWizUpgradeCommentName and CRLF to Comment
{$ENDIF}
            URL := '';
            if WizOptions.CurrentLangID = 2052 then // �������Ķ���������
              URL := Ini.ReadString(Strings[I], csURLCN, '');

            if URL = '' then  // û���������������ͨ����
              URL := Ini.ReadString(Strings[I], csURL, '');
            BetaVersion := Ini.ReadBool(Strings[I], csBetaVersion, False);
          end;
        {$IFDEF DEBUG}
          CnDebugger.LogObject(Item);
        {$ENDIF}
        except
          FreeAndNil(Item);
        end;
      end;
    end;
  finally
    if Assigned(Ini) then Ini.Free;
    if Assigned(Strings) then Strings.Free;
    Result := FUpgradeCollection.Count > 0;
  end;
end;

procedure TCnWizUpgradeThread.CheckUpgrade;
var
  I: Integer;
  
  function GetBuildNo(const VerStr: string): Integer;
  var
    s, s1: string;
    I: Integer;
  begin
    Result := 0;
    with TStringList.Create do
    try
      Text := StringReplace(VerStr, '.', CRLF, [rfReplaceAll]);
      if Count = 4 then
      begin
        s := Trim(Strings[3]);
        s1 := '';
        for I := 1 to Length(s) do
          if CharInSet(s[I], ['0'..'9']) then
            s1 := s1 + s[I]
          else
            Break;
        Result := StrToIntDef(s1, 0);
      end;
    finally
      Free;
    end;   
  end;
begin
  // �������������⵼������ʧ��ʱ����
  if FUpgradeCollection.Count = 0 then Exit;
  // �����������
  if (FUpgradeCollection[0].Date > WizOptions.BuildDate) or
    (GetBuildNo(FUpgradeCollection[0].Version) > GetBuildNo(SCnWizardVersion)) then
  begin
    // ɾ���ɰ汾��¼
    for I := FUpgradeCollection.Count - 1 downto 1 do
      if GetBuildNo(FUpgradeCollection[I].Version) <= GetBuildNo(SCnWizardVersion) then
        FUpgradeCollection.Delete(I);

    if not FUserCheck then
    begin
      // ����������ʾ
      if (WizOptions.UpgradeStyle = usDisabled) or (WizOptions.UpgradeStyle =
        usUserDefine) and (WizOptions.UpgradeContent = []) then
        Exit;

      // ɾ�����µĲ��԰汾��¼
      if WizOptions.UpgradeReleaseOnly then
        while FUpgradeCollection.Count > 0 do
          if FUpgradeCollection.Items[0].BetaVersion then
            FUpgradeCollection.Delete(0)
          else
            Break;

      // ɾ���������û��������ݲ����ļ�¼
      if WizOptions.UpgradeStyle = usUserDefine then
        while FUpgradeCollection.Count > 0 do
          if (ucNewFeature in WizOptions.UpgradeContent) and
            (FUpgradeCollection.Items[0].FNewFeature) or
            (ucBigBugFixed in WizOptions.UpgradeContent) and
            (FUpgradeCollection.Items[0].FBigBugFixed) then
            Break
          else
            FUpgradeCollection.Delete(0);

      // �ϴ���ʾ��û���µĸ���
      if (FUpgradeCollection.Count <= 0) or (Trunc(FUpgradeCollection.Items[0].Date)
        <= Trunc(WizOptions.UpgradeLastDate)) then
        Exit;
    end;
  end
  else
  begin
    if FUserCheck and QueryDlg(SCnWizNoUpgrade) then
      OpenUrl(WizOptions.NightlyBuildURL);
    Exit;
  end;

  // ��ʾ������ʾ����
  if FUpgradeCollection.Count > 0 then
  begin
    FForm := TCnWizUpgradeForm.Create(Application.MainForm);
    FForm.FCollection.Assign(FUpgradeCollection);
    FForm.Show;
  end;
end;

{ TCnWizUpgradeForm }

procedure TCnWizUpgradeForm.FormCreate(Sender: TObject);
begin
  FCollection := TCnWizUpgradeCollection.Create;
  ShowHint := WizOptions.ShowHint;

  if not NeedManuallyUnicode then
  begin
    FMemo := TMemo.Create(Self);

    // Memo
    FMemo.Name := 'Memo';
    FMemo.Parent := Self;
    FMemo.Left := 48;
    FMemo.Top := 24;
    FMemo.Width := 391;
    FMemo.Height := 208;
    FMemo.Align := alClient;
    FMemo.Color := clInfoBk;
    FMemo.ReadOnly := True;
    FMemo.ScrollBars := ssBoth;
    FMemo.TabOrder := 2;
    FMemo.WordWrap := False;
    FMemo.Lines.Clear;
  end
  else
  begin
    FPanel := TPanel.Create(Self);
    FLabelContent := TCnWideLabel.Create(Self);

    // FPanel
    FPanel.Name := 'FPanel';
    FPanel.Parent := Self;
    FPanel.Left := 48;
    FPanel.Top := 24;
    FPanel.Width := 391;
    FPanel.Height := 208;
    FPanel.BevelOuter := bvLowered;
    FPanel.Color := clInfoBk;
    FPanel.TabOrder := 4;
    FPanel.Caption := '';

    // FLabelContent
    FLabelContent.Name := 'FLabelContent';
    FLabelContent.Parent := FPanel;
    FLabelContent.Left := 3;
    FLabelContent.Top := 3;
    FLabelContent.Width := 387;
    FLabelContent.Height := 204;
    FLabelContent.Align := alClient;
    FLabelContent.Caption := '';
  end;
end;

procedure TCnWizUpgradeForm.FormShow(Sender: TObject);
var
  I: Integer;
  S: string;
  W, T: WideString;

  function AddLineCRLF(const Src: Widestring; const Subfix: WideString): WideString;
  begin
    if Subfix = '' then
      Result := Src + #13#10
    else
      Result := Src + Subfix + #13#10;
  end;

begin
  for I := 0 to FCollection.Count - 1 do
  begin
    if not NeedManuallyUnicode then
    begin
      with FMemo.Lines do
      begin
        S := Format(SCnWizUpgradeVersion, [FCollection.Items[I].Version,
          CnDateToStr(FCollection.Items[I].Date)]);
        Add(S);
        Add(GetLine('-', Length(S)));
        Add(FCollection.Items[I].Comment);
        Add('');
        Add('URL: ' + FCollection.Items[I].URL);
        if I < FCollection.Count - 1 then
        begin
          Add('');
          Add('');
        end;
      end;
    end
    else
    begin
      T := '';
      W := Format(SCnWizUpgradeVersion, [FCollection.Items[I].Version,
          CnDateToStr(FCollection.Items[I].Date)]);
      T := AddLineCRLF(T, W);
      T := AddLineCRLF(T, GetLine('-', Length(W)));
      T := AddLineCRLF(T, FCollection.Items[I].WideComment);
      T := AddLineCRLF(T, '');
      T := AddLineCRLF(T, 'URL: ' + FCollection.Items[I].URL);
      if I < FCollection.Count - 1 then
      begin
        T := AddLineCRLF(T, '');
        T := AddLineCRLF(T, '');
      end;

      FLabelContent.Caption := FLabelContent.Caption + T;
    end;
  end;
  cbNoHint.Checked := WizOptions.ReadBool(SCnUpgradeSection, csNoHint, True);
end;

procedure TCnWizUpgradeForm.FormClose(Sender: TObject;
  var Action: TCloseAction);
begin
  Action := caFree;
end;

procedure TCnWizUpgradeForm.FormDestroy(Sender: TObject);
begin
  FCollection.Free;
  FForm := nil;
end;

procedure TCnWizUpgradeForm.btnDownloadClick(Sender: TObject);
begin
  RunFile(FCollection.Items[0].URL);
  Close;
end;

procedure TCnWizUpgradeForm.btnHelpClick(Sender: TObject);
begin
  ShowFormHelp;
end;

function TCnWizUpgradeForm.GetHelpTopic: string;
begin
  Result := 'CnWizUpgrade';
end;

procedure TCnWizUpgradeForm.btnCloseClick(Sender: TObject);
begin
  if cbNoHint.Checked then
    WizOptions.UpgradeLastDate := FCollection.Items[0].Date;
  Close;
end;

procedure TCnWizUpgradeForm.cbNoHintClick(Sender: TObject);
begin
  WizOptions.WriteBool(SCnUpgradeSection, csNoHint, cbNoHint.Checked);
end;

function TCnWizUpgradeForm.NeedManuallyUnicode: Boolean;
begin
  Result := False;
{$IFNDEF UNICODE}
  // �� UNICODE �������£���ǰ������Ӣ��ҵ�ǰ CnWizards ������CHS/CHT ʱ
  if CodePageOnlySupportsEnglish and ((WizOptions.CurrentLangID = 2052)
    or (WizOptions.CurrentLangID = 1028)) then
    Result := True;
{$ENDIF}
end;

initialization

finalization
{$IFDEF DEBUG}
  CnDebugger.LogEnter('CnWizUpgradeFrm finalization.');
{$ENDIF}
  if Assigned(FThread) then
    try
      // �����ǰ����ִ�� HTTP ���������ܲ��������˳����˴�ǿ���˳��߳�
      TerminateThread(FThread.Handle, 0);
      if Assigned(FThread) then
        FreeAndNil(FThread);
    except
      ;
    end;
  if FForm <> nil then
    FForm.Free;

{$IFDEF DEBUG}
  CnDebugger.LogLeave('CnWizUpgradeFrm finalization.');
{$ENDIF}
end.

