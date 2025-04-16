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

unit CnAICoderWizard;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�AI ��������ר�ҵ�Ԫ
* ��Ԫ���ߣ�CnPack ������
* ��    ע��
* ����ƽ̨��PWin7 + Delphi 5
* ���ݲ��ԣ�PWin7/10/11 + Delphi + C++Builder
* �� �� �����ô����е��ַ����ݲ�֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2024.04.29 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNAICODERWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ToolsAPI, IniFiles,ComCtrls, StdCtrls, CnConsts, CnWizClasses, CnWizUtils,
  CnWizConsts, CnCommon, CnAICoderConfig, CnThreadPool, CnAICoderEngine,
  CnFrmAICoderOption, CnWizMultiLang;

type
  TCnAICoderConfigForm = class(TCnTranslateForm)
    lblActiveEngine: TLabel;
    cbbActiveEngine: TComboBox;
    pgcAI: TPageControl;
    btnOK: TButton;
    btnCancel: TButton;
    btnHelp: TButton;
    chkProxy: TCheckBox;
    edtProxy: TEdit;
    lblTimeout: TLabel;
    edtTimeout: TEdit;
    udTimeout: TUpDown;
    procedure cbbActiveEngineChange(Sender: TObject);
    procedure btnHelpClick(Sender: TObject);
  private
    FTabsheets: array of TTabSheet;
    FOptionFrames: array of TCnAICoderOptionFrame;
  protected
    function GetHelpTopic: string; override;
  public
    procedure LoadFromOptions;
    procedure SaveToOptions;
  end;
 
//==============================================================================
// AI ���������Ӳ˵�ר��
//==============================================================================

{ TCnAICoderWizard }

  TCnAICoderWizard = class(TCnSubMenuWizard)
  private
    FIdExplainCode: Integer;
    FIdReviewCode: Integer;
    FIdGenTestCase: Integer;
    FIdShowChatWindow: Integer;
    FIdConfig: Integer;
    function ValidateAIEngines: Boolean;
    {* ���ø�������ǰ��� AI ���漰����}
    procedure EnsureChatWindowVisible;
    {* ȷ������ ChatWindow ���� Visible Ϊ True �������� Parent �� Visible ȫΪ True
      ��ȷ�����촰�ڿɼ�}
  protected
    function GetHasConfig: Boolean; override;
    procedure SubActionExecute(Index: Integer); override;
    procedure SubActionUpdate(Index: Integer); override;

    procedure SetActive(Value: Boolean); override;
  public
    constructor Create; override;
    destructor Destroy; override;

    procedure ForCodeAnswer(StreamMode, Partly, Success, IsStreamEnd: Boolean;
      SendId: Integer; const Answer: string; ErrorCode: Cardinal; Tag: TObject);
    {* �������ֵĻص�}
    procedure ForCodeGen(StreamMode, Partly, Success, IsStreamEnd: Boolean;
      SendId: Integer; const Answer: string; ErrorCode: Cardinal; Tag: TObject);
    {* ���ش���Ļص�}

    procedure AcquireSubActions; override;
    function GetState: TWizardState; override;
    procedure Config; override;
    procedure LoadSettings(Ini: TCustomIniFile); override;
    procedure SaveSettings(Ini: TCustomIniFile); override;
    class procedure GetWizardInfo(var Name, Author, Email, Comment: string); override;
    function GetCaption: string; override;
    function GetHint: string; override;
  end;

{$ENDIF CNWIZARDS_CNAICODERWIZARD}

implementation

{$IFDEF CNWIZARDS_CNAICODERWIZARD}

{$R *.DFM}

uses
  CnWizOptions, CnAICoderNetClient, CnAICoderChatFrm, CnChatBox, CnWizIdeDock
  {$IFDEF DEBUG} , CnDebug {$ENDIF};

const
  MSG_WAITING = '...';
  csAICoderChatForm = 'CnAICoderChatForm';

//==============================================================================
// AI ��������˵�ר��
//==============================================================================

{ TCnAICoderWizard }

procedure TCnAICoderWizard.Config;
begin
  with TCnAICoderConfigForm.Create(nil) do
  begin
    LoadFromOptions;
    if ShowModal = mrOK then
    begin
      SaveToOptions;

      DoSaveSettings;

      if CnAICoderChatForm <> nil then
        CnAICoderChatForm.NotifySettingChanged;
    end;
    Free;
  end;
end;

constructor TCnAICoderWizard.Create;
begin
  inherited;
  IdeDockManager.RegisterDockableForm(TCnAICoderChatForm, CnAICoderChatForm,
    csAICoderChatForm);
end;

destructor TCnAICoderWizard.Destroy;
begin
  IdeDockManager.UnRegisterDockableForm(CnAICoderChatForm, csAICoderChatForm);
  FreeAndNil(CnAICoderChatForm);
  inherited;
end;

// �������ظ÷����������Ӳ˵�ר����
procedure TCnAICoderWizard.AcquireSubActions;
begin
  FIdExplainCode := RegisterASubAction(SCnAICoderWizardExplainCode,
    SCnAICoderWizardExplainCodeCaption, 0,
    SCnAICoderWizardExplainCodeHint, SCnAICoderWizardExplainCode);

  FIdReviewCode := RegisterASubAction(SCnAICoderWizardReviewCode,
    SCnAICoderWizardReviewCodeCaption, 0,
    SCnAICoderWizardReviewCodeHint, SCnAICoderWizardReviewCode);

  FIdGenTestCase := RegisterASubAction(SCnAICoderWizardGenTestCase,
    SCnAICoderWizardGenTestCaseCaption, 0,
    SCnAICoderWizardGenTestCaseHint, SCnAICoderWizardGenTestCase);

  // �����ָ��˵�
  AddSepMenu;

  FIdShowChatWindow := RegisterASubAction(SCnAICoderWizardChatWindow,
    SCnAICoderWizardChatWindowCaption, 0,
    SCnAICoderWizardChatWindowHint, SCnAICoderWizardChatWindow);

  FIdConfig := RegisterASubAction(SCnAICoderWizardConfig,
    SCnAICoderWizardConfigCaption, 0, SCnAICoderWizardConfigHint, SCnAICoderWizardConfig);
end;

function TCnAICoderWizard.GetCaption: string;
begin
  Result := SCnAICoderWizardMenuCaption;
end;

function TCnAICoderWizard.GetHasConfig: Boolean;
begin
  Result := True;
end;

function TCnAICoderWizard.GetHint: string;
begin
  Result := SCnAICoderWizardMenuHint;
end;

function TCnAICoderWizard.GetState: TWizardState;
begin
  Result := [wsEnabled];
end;

class procedure TCnAICoderWizard.GetWizardInfo(var Name, Author, Email, Comment: string);
begin
  Name := SCnAICoderWizardName;
  Author := SCnPack_LiuXiao;
  Email := SCnPack_LiuXiaoEmail;
  Comment := SCnAICoderWizardComment;
end;

procedure TCnAICoderWizard.LoadSettings(Ini: TCustomIniFile);
begin
  CnAIEngineManager.LoadFromWizOptions;

  // ������Ҫ���ֶ����ô洢�Ļ��������
  CnAIEngineManager.CurrentEngineName := CnAIEngineOptionManager.ActiveEngine;

{$IFDEF DEBUG}
  CnDebugger.LogFmt('CnAIEngineOptionManager Load %d Options.', [CnAIEngineOptionManager.OptionCount]);
{$ENDIF}
end;

procedure TCnAICoderWizard.SaveSettings(Ini: TCustomIniFile);
begin
  CnAIEngineManager.SaveToWizOptions;
end;

procedure TCnAICoderWizard.SubActionExecute(Index: Integer);
var
  S: string;
  Msg: TCnChatMessage;
begin
  if not Active then
    Exit;

  if Index = FIdConfig then
    Config
  else if Index = FIdShowChatWindow then
  begin
    EnsureChatWindowVisible;
  end
  else
  begin
    if not ValidateAIEngines then
    begin
      Config;
      Exit;
    end;

    if (Index = FIdExplainCode) or (Index = FIdReviewCode) then
    begin
      S := CnOtaGetCurrentSelection;
      if Trim(S) <> '' then
      begin
        EnsureChatWindowVisible;
        Msg := CnAICoderChatForm.ChatBox.Items.AddMessage;
        Msg.From := CnAIEngineManager.CurrentEngineName;
        Msg.FromType := cmtYou;
        Msg.Text := MSG_WAITING;
        Msg.Waiting := True;

        if Index = FIdExplainCode then
          CnAIEngineManager.CurrentEngine.AskAIEngineForCode(S, Msg, artExplainCode, ForCodeAnswer)
        else
          CnAIEngineManager.CurrentEngine.AskAIEngineForCode(S, Msg, artReviewCode, ForCodeAnswer);
      end;
    end
    else if Index = FIdGenTestCase then
    begin
      S := CnOtaGetCurrentSelection;
      if Trim(S) <> '' then
      begin
        EnsureChatWindowVisible;

        Msg := CnAICoderChatForm.ChatBox.Items.AddMessage;
        Msg.From := CnAIEngineManager.CurrentEngineName;
        Msg.FromType := cmtYou;
        Msg.Text := '...';
        Msg.Waiting := True;

        CnAIEngineManager.CurrentEngine.AskAIEngineForCode(S, Msg, artGenTestCase, ForCodeGen);
      end;
    end;
  end;
end;

procedure TCnAICoderWizard.SubActionUpdate(Index: Integer);
begin
  if (Index = FIdConfig) or (Index = FIdShowChatWindow) then
    SubActions[Index].Enabled := Active
  else
    SubActions[Index].Enabled := Active and (CnOtaGetCurrentSelection <> '');
end;

procedure TCnAICoderWizard.SetActive(Value: Boolean);
var
  Old: Boolean;
begin
  Old := Active;
  inherited;
  if Old <> Active then
  begin
    if Active then
    begin
      IdeDockManager.RegisterDockableForm(TCnAICoderChatForm, CnAICoderChatForm,
        'CnAICoderChatForm');
    end
    else
    begin
      IdeDockManager.UnRegisterDockableForm(CnAICoderChatForm, 'CnAICoderChatForm');
      FreeAndNil(CnAICoderChatForm);
    end;
  end;
end;

function TCnAICoderWizard.ValidateAIEngines: Boolean;
begin
  Result := False;
  if (CnAIEngineManager.CurrentEngine = nil) or
    (CnAIEngineManager.CurrentEngine.Option = nil) then
  begin
    ErrorDlg(SCnAICoderWizardErrorNoEngine);
    Exit;
  end;
  if (Trim(CnAIEngineManager.CurrentEngine.Option.URL) = '') then
  begin
    ErrorDlg(Format(SCnAICoderWizardErrorURLFmt, [CnAIEngineManager.CurrentEngine.EngineName]));
    Exit;
  end;
  if CnAIEngineManager.CurrentEngine.NeedApiKey and (Trim(CnAIEngineManager.CurrentEngine.Option.ApiKey) = '') then
  begin
    ErrorDlg(Format(SCnAICoderWizardErrorAPIKeyFmt, [CnAIEngineManager.CurrentEngine.EngineName]));
    Exit;
  end;

  Result := True;
end;

procedure TCnAICoderConfigForm.LoadFromOptions;
var
  I, J, C: Integer;
  SL: TStringList;
  Eng: TCnAIBaseEngine;
begin
  chkProxy.Checked := CnAIEngineOptionManager.UseProxy;
  edtProxy.Text := CnAIEngineOptionManager.ProxyServer;

  cbbActiveEngine.Items.Clear;
  for I := 0 to CnAIEngineManager.EngineCount - 1 do
    cbbActiveEngine.Items.Add(CnAIEngineManager.Engines[I].EngineName);

  cbbActiveEngine.ItemIndex := CnAIEngineManager.CurrentIndex;

  udTimeout.Position := CnAIEngineOptionManager.TimeoutSec;

  // ��ÿ�� Options ����һ�� Tab��ÿ�� Tab ����һ�� Frame���� Frame ��Ķ����� Option ����
  SetLength(FTabsheets, CnAIEngineOptionManager.OptionCount);
  SetLength(FOptionFrames, CnAIEngineOptionManager.OptionCount);

  SL := TStringList.Create;
  try
    for I := 0 to CnAIEngineOptionManager.OptionCount - 1 do
    begin
      Eng := CnAIEngineManager.GetEngineByOption(CnAIEngineOptionManager.Options[I]);

      // ��ÿ�� Options ����һ�� Tab
      FTabsheets[I] := TTabSheet.Create(pgcAI);
      FTabsheets[I].Caption := CnAIEngineOptionManager.Options[I].EngineName + Format(' (&%d)', [I]);
      FTabsheets[I].PageControl := pgcAI;

      // ��ÿ�� Tab ����һ�� Frame
      FOptionFrames[I] := TCnAICoderOptionFrame.Create(FTabsheets[I]);
      FOptionFrames[I].Engine := Eng;
      FOptionFrames[I].Name := 'CnAICoderOptionFrame' + IntToStr(I);
      FOptionFrames[I].Parent := FTabsheets[I];
      FOptionFrames[I].Top := 0;
      FOptionFrames[I].Left := 0;
      FOptionFrames[I].Align := alClient;

      // ��ÿ�� Frame ��Ķ����� Option ����
      FOptionFrames[I].LoadFromAnOption(CnAIEngineOptionManager.Options[I]);

      // �������Ҫ APIKey �ͽ���
      if (Eng <> nil) and not Eng.NeedApiKey then
      begin
        FOptionFrames[I].lblAPIKey.Enabled := False;
        FOptionFrames[I].edtAPIKey.Enabled := False;
      end;

      // �Ѹ� Option ��Ķ���������� Frame ʵ����������ֵ
      C := CnAIEngineOptionManager.Options[I].GetExtraOptionCount;
      if C > 0 then
      begin
        for J := 0 to C - 1 do
        begin
          FOptionFrames[I].RegisterExtraOption(CnAIEngineOptionManager.Options[I],
            CnAIEngineOptionManager.Options[I].GetExtraOptionName(J),
            CnAIEngineOptionManager.Options[I].GetExtraOptionType(J));
        end;
        FOptionFrames[I].BuildExtraOptionElements;
      end;

      SL.Clear;
      ExtractStrings([','], [' '], PChar(CnAIEngineOptionManager.Options[I].ModelList), SL);
      if SL.Count > 0 then
        FOptionFrames[I].cbbModel.Items.Assign(SL);

      // �еĻ�����ַ��������ϣ��������
      if CnAIEngineOptionManager.Options[I].WebAddress <> '' then
        FOptionFrames[I].WebAddr := CnAIEngineOptionManager.Options[I].WebAddress
      else
        FOptionFrames[I].lblApply.Visible := False;
    end;
  finally
    SL.Free;
  end;
  pgcAI.ActivePageIndex := CnAIEngineManager.CurrentIndex;
end;

procedure TCnAICoderConfigForm.SaveToOptions;
var
  I: Integer;
begin
  for I := 0 to Length(FOptionFrames) - 1 do
  begin
    // ���׼����
    FOptionFrames[I].SaveToAnOption(CnAIEngineOptionManager.Options[I]);

    // ���������
    FOptionFrames[I].SaveExtraOptions;
  end;

  CnAIEngineOptionManager.ActiveEngine := cbbActiveEngine.Text;
  CnAIEngineManager.CurrentEngineName := CnAIEngineOptionManager.ActiveEngine;

  CnAIEngineOptionManager.TimeoutSec := udTimeout.Position;

  CnAIEngineOptionManager.UseProxy := chkProxy.Checked;
  CnAIEngineOptionManager.ProxyServer := edtProxy.Text;
end;

procedure TCnAICoderWizard.ForCodeAnswer(StreamMode, Partly, Success, IsStreamEnd: Boolean;
  SendId: Integer; const Answer: string; ErrorCode: Cardinal; Tag: TObject);
begin
  EnsureChatWindowVisible;

  if (Tag <> nil) and (Tag is TCnChatMessage) then
  begin
    TCnChatMessage(Tag).Waiting := False;
    if Success then
    begin
      if Partly and (TCnChatMessage(Tag).Text <> MSG_WAITING) then
        TCnChatMessage(Tag).Text := TCnChatMessage(Tag).Text + Answer
      else
        TCnChatMessage(Tag).Text := Answer;
    end
    else
      TCnChatMessage(Tag).Text := Format('%d %s', [ErrorCode, Answer]);
  end
  else
  begin
    if Success then
      CnAICoderChatForm.AddMessage(Answer, CnAIEngineManager.CurrentEngineName)
    else
      CnAICoderChatForm.AddMessage(Format('%d %s', [ErrorCode, Answer]), CnAIEngineManager.CurrentEngineName);
  end;
end;

procedure TCnAICoderWizard.ForCodeGen(StreamMode, Partly, Success, IsStreamEnd: Boolean;
  SendId: Integer; const Answer: string; ErrorCode: Cardinal; Tag: TObject);
var
  S: string;
begin
  if (Tag <> nil) and (Tag is TCnChatMessage) then
  begin
    TCnChatMessage(Tag).Waiting := False;
    if Success then
    begin
      if Partly and (TCnChatMessage(Tag).Text <> MSG_WAITING)then
        TCnChatMessage(Tag).Text := TCnChatMessage(Tag).Text + Answer
      else
        TCnChatMessage(Tag).Text := Answer;

      // ��������������
      if not Partly or IsStreamEnd then
      begin
        S := TCnAICoderChatForm.ExtractCode(TCnChatMessage(Tag));
        if S <> '' then
        begin
          // �ж�����ѡ���������⸲��ѡ��������
          if CnOtaGetCurrentSelection <> '' then // ȡ��ѡ�񣬲����ƹ��
            CnOtaDeSelection(True);

          CnOtaInsertTextIntoEditor(#13#10 + S + #13#10);
        end;
      end;
    end
    else
    begin
      EnsureChatWindowVisible;
      TCnChatMessage(Tag).Text := Format('%d %s', [ErrorCode, Answer]);
    end;
  end
  else
  begin
    EnsureChatWindowVisible;
    if Success then
      CnAICoderChatForm.AddMessage(Answer, CnAIEngineManager.CurrentEngineName)
    else
      CnAICoderChatForm.AddMessage(Format('%d %s', [ErrorCode, Answer]), CnAIEngineManager.CurrentEngineName);
  end;
end;

procedure TCnAICoderWizard.EnsureChatWindowVisible;
begin
  if CnAICoderChatForm = nil then
  begin
    CnAICoderChatForm := TCnAICoderChatForm.Create(nil);
    CnAICoderChatForm.Wizard := Self;
  end;

  CnAICoderChatForm.VisibleWithParent := True;
  CnAICoderChatForm.BringToFront;
end;

procedure TCnAICoderConfigForm.cbbActiveEngineChange(Sender: TObject);
begin
  if (cbbActiveEngine.ItemIndex >= 0) and (cbbActiveEngine.ItemIndex < pgcAI.PageCount) then
    pgcAI.ActivePageIndex := cbbActiveEngine.ItemIndex;
end;

procedure TCnAICoderConfigForm.btnHelpClick(Sender: TObject);
begin
  ShowFormHelp;
end;

function TCnAICoderConfigForm.GetHelpTopic: string;
begin
  Result := 'CnAICoderWizard';
end;

initialization
  RegisterCnWizard(TCnAICoderWizard); // ע��ר��

{$ENDIF CNWIZARDS_CNAICODERWIZARD}
end.
