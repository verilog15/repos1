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

unit CnPrefixEditFrm;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ����ǰ׺ר������������嵥Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע�����ǰ׺ר������������嵥Ԫ
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2003.04.26 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNPREFIXWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls, CnWizConsts, CnCommon, CnWizUtils, CnWizMultiLang,
  Buttons;

type

{ TCnPrefixEditForm }

  TCnPrefixEditForm = class(TCnTranslateForm)
    gbEdit: TGroupBox;
    lblFormName: TLabel;
    bvl1: TBevel;
    lbl1: TLabel;
    lbl2: TLabel;
    lbl3: TLabel;
    edtName: TEdit;
    btnOK: TButton;
    btnCancel: TButton;
    btnHelp: TButton;
    cbNeverDisp: TCheckBox;
    cbIgnoreComp: TCheckBox;
    btnPrefix: TButton;
    img1: TImage;
    edtOldName: TEdit;
    lbl4: TLabel;
    lblClassName: TLabel;
    lbl5: TLabel;
    lblText: TLabel;
    btnClassName: TSpeedButton;
    chkDisablePrefix: TCheckBox;
    procedure btnOKClick(Sender: TObject);
    procedure btnHelpClick(Sender: TObject);
    procedure btnPrefixClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure edtNameKeyPress(Sender: TObject; var Key: Char);
    procedure btnClassNameClick(Sender: TObject);
  private
    FPrefix: string;
    FRootName: string;
    FUseUnderLine: Boolean;
    FComponentClass: string;
    procedure SetEditSel(Sender: TObject);
  protected
    function GetHelpTopic: string; override;
  public

  end;

// ��ʾ�Ի���ȡ���µ�������ơ�RootName ��Ϊ��ʱ��ʾ�� Form ������
function GetNewComponentName(const FormName, ComponentClass, ComponentText,
  OldName: string; var Prefix, NewName: string; HideMode: Boolean;
  var IgnoreComp, AutoPopSuggestDlg, WizardActive: Boolean; UseUnderLine: Boolean;
  const RootName: string = ''; AWizard: TObject = nil): Boolean;

{$ENDIF CNWIZARDS_CNPREFIXWIZARD}

implementation

{$IFDEF CNWIZARDS_CNPREFIXWIZARD}

uses
  CnPrefixNewFrm, CnPrefixWizard, CnWizNotifier {$IFDEF DEBUG}, CnDebug {$ENDIF};

{$R *.DFM}

{ TCnPrefixEditForm }

// ȡ���µ��������
function GetNewComponentName(const FormName, ComponentClass, ComponentText,
  OldName: string; var Prefix, NewName: string; HideMode: Boolean;
  var IgnoreComp, AutoPopSuggestDlg, WizardActive: Boolean; UseUnderLine: Boolean;
  const RootName: string; AWizard: TObject): Boolean;
var
  Wizard: TCnPrefixWizard;
  OldWidth, OldHeight: Integer;
begin
  Result := False;
  if not (AWizard is TCnPrefixWizard) then
    Exit;

  Wizard := AWizard as TCnPrefixWizard;
  with TCnPrefixEditForm.Create(nil) do
  try
    // ���ر����δ���ųߴ粢����
    if Wizard.EditDialogWidth > 0 then
      Width := CalcIntEnlargedValue(Wizard.EditDialogWidth);
    if  Wizard.EditDialogHeight > 0 then
      Height := CalcIntEnlargedValue(Wizard.EditDialogHeight);

    lblFormName.Caption := FormName;
    lblClassName.Caption := ComponentClass;
    lblText.Caption := ComponentText;
    FUseUnderLine := UseUnderLine;
    FPrefix := Prefix;
    FRootName := RootName;
    FComponentClass := ComponentClass;
    edtOldName.Text := OldName;
    edtName.Text := NewName;
    cbNeverDisp.Checked := not AutoPopSuggestDlg;
    chkDisablePrefix.Checked := not WizardActive;
    if HideMode then
    begin
      cbIgnoreComp.Visible := False;
      cbNeverDisp.Visible := False;
      chkDisablePrefix.Visible := False;
    end;

    OldWidth := Width;
    OldHeight := Height;
    Result := ShowModal = mrOk;

    Prefix := FPrefix;
    NewName := edtName.Text;
    IgnoreComp := cbIgnoreComp.Checked;
    AutoPopSuggestDlg := not cbNeverDisp.Checked;
    WizardActive := not chkDisablePrefix.Checked;

    // ����ߴ�ı��ˣ�����δ���ź�ĳߴ�
    if (Width <> OldWidth) or (Height <> OldHeight) then
    begin
      Wizard.EditDialogWidth := CalcIntUnEnlargedValue(Width);
      Wizard.EditDialogHeight := CalcIntUnEnlargedValue(Height);
{$IFDEF DEBUG}
      CnDebugger.LogFmt('GetNewComponentName from Prefix Dialog. Save Width %d, Height %d,',
        [Wizard.EditDialogWidth, Wizard.EditDialogHeight]);
{$ENDIF}
    end;

    if not WizardActive then
      Result := False;
  finally
    Free;
  end;
end;

procedure TCnPrefixEditForm.FormShow(Sender: TObject);
begin
  CnWizNotifierServices.ExecuteOnApplicationIdle(SetEditSel);
end;

procedure TCnPrefixEditForm.btnOKClick(Sender: TObject);
begin
  if IsValidIdent(edtName.Text) then
    ModalResult := mrOk
  else
    ErrorDlg(SCnPrefixNameError);
end;

procedure TCnPrefixEditForm.btnPrefixClick(Sender: TObject);
var
  B1, B2: Boolean;
  OldPrefix: string;
begin
  OldPrefix := FPrefix;
  if GetNewComponentPrefix(FComponentClass, FPrefix, True, B1, B2, FRootName) then
    if Pos(OldPrefix, edtName.Text) = 1 then
      edtName.Text := StringReplace(edtName.Text, OldPrefix, FPrefix, []);

  SetEditSel(nil);
end;

procedure TCnPrefixEditForm.SetEditSel(Sender: TObject);
begin
  edtName.SetFocus;
  if Self.FUseUnderLine then
  begin
    edtName.SelStart := Length(FPrefix) + 1;
    edtName.SelLength := Length(edtName.Text) - Length(FPrefix) - 1;
  end
  else
  begin
    edtName.SelStart := Length(FPrefix);
    edtName.SelLength := Length(edtName.Text) - Length(FPrefix);
  end;
end;

procedure TCnPrefixEditForm.btnHelpClick(Sender: TObject);
begin
  ShowFormHelp;
end;

function TCnPrefixEditForm.GetHelpTopic: string;
begin
  Result := 'CnPrefixEditForm';
end;

procedure TCnPrefixEditForm.edtNameKeyPress(Sender: TObject;
  var Key: Char);
const
  Chars = ['A'..'Z', 'a'..'z', '_', '0'..'9', #03, #08, #22, #24, #26]; // Ctrl+C/V/X/Z
begin
  if not CharInSet(Key, Chars) and not IsValidIdent('A' + Key) then
    Key := #0;
end;

procedure TCnPrefixEditForm.btnClassNameClick(Sender: TObject);
begin
  edtName.Text := RemoveClassPrefix(lblClassName.Caption);
end;

{$ENDIF CNWIZARDS_CNPREFIXWIZARD}
end.
