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

unit CnWizAboutFrm;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ����ڴ��嵥Ԫ
* ��Ԫ���ߣ�CnPack������
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2021.07.26 V1.2
*               ������������
*           2003.03.10 V1.1
*               ������ͼƬ
*           2002.09.28 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, StdCtrls, CnConsts, CnWizFeedbackFrm, CnWizMultiLang, CnLangMgr,
  CnWaterImage, CnWizIdeUtils;

type
  TCnWizAboutForm = class(TCnTranslateForm)
    Bevel1: TBevel;
    Label2: TLabel;
    Label4: TLabel;
    btnOK: TButton;
    lblWeb: TLabel;
    lblEmail: TLabel;
    lblVersion: TLabel;
    lblBbs: TLabel;
    Bevel2: TBevel;
    Label3: TLabel;
    btnReport: TButton;
    Panel1: TPanel;
    btnLicense: TButton;
    tmr1: TTimer;
    CnWaterImage1: TCnWaterImage;
    imgDonation: TImage;
    edtVer: TEdit;
    lblSource: TLabel;
    procedure lblWebClick(Sender: TObject);
    procedure lblEmailClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure lblBbsClick(Sender: TObject);
    procedure btnReportClick(Sender: TObject);
    procedure btnLicenseClick(Sender: TObject);
    procedure imgDonationClick(Sender: TObject);
    procedure lblSourceClick(Sender: TObject);
    procedure CnWaterImage1MouseUp(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
  private
    procedure DbgEditKeyPress(Sender: TObject; var Key: Char);
  protected
    function GetHelpTopic: string; override;
  public
  end;

// ��ʾ���ڴ���
procedure ShowCnWizAboutForm;

implementation

uses
  CnCommon, CnWizConsts, CnWizOptions, CnWizManager;

{$R *.DFM}

var
  DbgFrm: TForm = nil;

// ��ʾ���ڴ���
procedure ShowCnWizAboutForm;
begin
  with TCnWizAboutForm.Create(Application.MainForm) do
  try
    ShowHint := WizOptions.ShowHint;
    ShowModal;
  finally
    Free;
  end;
end;

{ TCnWizAboutForm }

procedure TCnWizAboutForm.FormCreate(Sender: TObject);
begin
  edtVer.Text := Format('%s %s.%s Build %s', [SCnVersion,
    SCnWizardMajorVersion, SCnWizardMinorVersion, SCnWizardBuildDate]);

{$IFDEF WIN64}
  Caption := Caption + ' (64) - ' + _CnExtractFileName(WizOptions.DllName);
{$ELSE}
  Caption := Caption + ' - ' + _CnExtractFileName(WizOptions.DllName);
{$ENDIF}
end;

procedure TCnWizAboutForm.lblWebClick(Sender: TObject);
begin
  OpenUrl(SCnPackUrl);
end;

procedure TCnWizAboutForm.lblEmailClick(Sender: TObject);
begin
  MailTo(SCnPackEmail, SCnWizMailSubject);
end;

procedure TCnWizAboutForm.lblBbsClick(Sender: TObject);
begin
  OpenUrl(SCnPackBbsUrl);
end;

procedure TCnWizAboutForm.lblSourceClick(Sender: TObject);
begin
  OpenUrl(SCnPackSourceUrl);
end;

procedure TCnWizAboutForm.imgDonationClick(Sender: TObject);
begin
  OpenUrl(SCnPackDonationUrl);
end;

procedure TCnWizAboutForm.btnReportClick(Sender: TObject);
begin
  ShowFeedbackForm;
  ModalResult := mrOk;
end;

procedure TCnWizAboutForm.btnLicenseClick(Sender: TObject);
begin
  ShowFormHelp;
end;

function TCnWizAboutForm.GetHelpTopic: string;
begin
  Result := 'License';
end;

procedure TCnWizAboutForm.DbgEditKeyPress(Sender: TObject; var Key: Char);
var
  List: TStrings;
  Memo: TMemo;
  Cmd: string;
begin
  if Key = #13 then
  begin
    if not (Sender is TEdit) then
      Exit;

    Memo := TMemo((Sender as TEdit).Tag);
    if Memo = nil then
      Exit;

    Cmd := Trim((Sender as TEdit).Text);
    if Cmd = '' then
      Exit;

    List := TStringList.Create;
    try
      CnWizardMgr.DispatchDebugComand(Cmd, List);
      Memo.Clear;
      Memo.Lines.AddStrings(List);
    finally
      List.Free;
    end;
    Key := #0;
  end;
end;

procedure TCnWizAboutForm.CnWaterImage1MouseUp(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var
  Edit: TEdit;
  Memo: TMemo;
begin
  if not (Button in [mbLeft, mbRight]) then
    Exit;

  // Ctrl/Shift/Alt ���ð���
  if not (ssShift in Shift) or not (ssCtrl in Shift) or not (ssAlt in Shift) then
    Exit;

  Close;

  if DbgFrm <> nil then
  begin
    DbgFrm.Show;
  end
  else
  begin
    DbgFrm := TForm.Create(Application);
    with DbgFrm do
    begin
      Width := IdeGetScaledPixelsFromOrigin(550);
      Height := IdeGetScaledPixelsFromOrigin(400);
      Position := poScreenCenter;
      BorderStyle := bsSizeToolWin;
      Caption := 'CnPack IDE Wizard Debug Command Window';
      BorderIcons := [biSystemMenu];
    end;

    Edit := TEdit.Create(DbgFrm);
    with Edit do
    begin
      Parent := DbgFrm;
      Align := alTop;
      OnKeyPress := DbgEditKeyPress;
    end;

    Memo := TMemo.Create(DbgFrm);
    with Memo do
    begin
      Parent := DbgFrm;
      Align := alClient;
      ReadOnly := True;
      ScrollBars := ssBoth;
      Text := '';
    end;

    Edit.Tag := Integer(Memo);
    DbgFrm.Show;
  end;
end;

end.
