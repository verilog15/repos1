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

unit CnIniFilerFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�Ini �ļ���д��Ԫ����
* ��Ԫ���ߣ�LiuXiao ��master@cnpack.org��
* ��    ע��Ini �ļ���д��Ԫ����
* ����ƽ̨��Windows 2000 + Delphi 5
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2003.12.07 V1.0
*               LiuXiao ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNINIFILERWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Buttons, CnWizMultiLang, CnCommon, CnWizConsts;

type
  TCnIniFilerForm = class(TCnTranslateForm)
    btnOK: TButton;
    btnCancel: TButton;
    btnHelp: TButton;
    grp1: TGroupBox;
    lblIni: TLabel;
    edtIniFile: TEdit;
    lblConstPrefix: TLabel;
    edtPrefix: TEdit;
    lblIniClassName: TLabel;
    edtClassName: TEdit;
    lblT: TLabel;
    dlgOpen: TOpenDialog;
    btnOpen: TSpeedButton;
    chkIsAllStr: TCheckBox;
    chkBool: TCheckBox;
    dlgSave: TSaveDialog;
    chkSectionMode: TCheckBox;
    procedure btnOpenClick(Sender: TObject);
    procedure btnHelpClick(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
    procedure chkIsAllStrClick(Sender: TObject);
  private
    function GetConstPrefix: string;
    procedure SetConstPrefix(const Value: string);
    function GetIniClassName: string;
    procedure SetIniClassName(const Value: string);
    function GetIniFileName: string;
    procedure SetIniFileName(const Value: string);
    function GetIsAllStr: Boolean;
    procedure SetIsAllStr(const Value: Boolean);
    function GetCheckBool: Boolean;
    procedure SetCheckBool(const Value: Boolean);
    function GetSectionMode: Boolean;
    procedure SetSectionMode(const Value: Boolean);
  protected
    function GetHelpTopic: string; override;
  public
    property ConstPrefix: string read GetConstPrefix write SetConstPrefix;
    property IniClassName: string read GetIniClassName write SetIniClassName;
    property IniFileName: string read GetIniFileName write SetIniFileName;
    property IsAllStr: Boolean read GetIsAllStr write SetIsAllStr;
    property CheckBool: Boolean read GetCheckBool write SetCheckBool;
    property SectionMode: Boolean read GetSectionMode write SetSectionMode;
  end;

{$ENDIF CNWIZARDS_CNINIFILERWIZARD}

implementation

{$IFDEF CNWIZARDS_CNINIFILERWIZARD}

{$R *.DFM}

function TCnIniFilerForm.GetConstPrefix: string;
begin
  Result := edtPrefix.Text;
end;

procedure TCnIniFilerForm.SetConstPrefix(const Value: string);
begin
  edtPrefix.Text := Value;
end;

function TCnIniFilerForm.GetIniClassName: string;
begin
  Result := edtClassName.Text;
end;

procedure TCnIniFilerForm.SetIniClassName(const Value: string);
begin
  edtClassName.Text := Value;
end;

procedure TCnIniFilerForm.btnOpenClick(Sender: TObject);
begin
  if dlgOpen.Execute then
    IniFileName := dlgOpen.FileName;
end;

function TCnIniFilerForm.GetIniFileName: string;
begin
  Result := edtIniFile.Text;
end;

procedure TCnIniFilerForm.SetIniFileName(const Value: string);
begin
  edtIniFile.Text := Value;
end;

procedure TCnIniFilerForm.btnHelpClick(Sender: TObject);
begin
  ShowFormHelp;
end;

procedure TCnIniFilerForm.FormCloseQuery(Sender: TObject;
  var CanClose: Boolean);
begin
  CanClose := True;
  if ModalResult = mrOK then
  begin
    if (IniFileName = '') or not FileExists(IniFileName) then
    begin
      ErrorDlg(SCnIniErrorNoFile);
      CanClose := False;
    end
    else if (IniClassName = '') or (Pos(' ', IniClassName) > 0) then
    begin
      ErrorDlg(SCnIniErrorClassName);
      CanClose := False;
    end
    else if (ConstPrefix = '') or (Pos(' ', ConstPrefix) > 0) then
    begin
      ErrorDlg(SCnIniErrorPrefix);
      CanClose := False;
    end;
  end;
end;

function TCnIniFilerForm.GetHelpTopic: string;
begin
  Result := 'CnIniFilerWizard';
end;

function TCnIniFilerForm.GetIsAllStr: Boolean;
begin
  Result := not Self.chkIsAllStr.Checked;
end;

procedure TCnIniFilerForm.SetIsAllStr(const Value: Boolean);
begin
  chkIsAllStr.Checked := not Value;
  if Assigned(chkIsAllStr.OnClick) then
    chkIsAllStr.OnClick(chkIsAllStr);
end;

function TCnIniFilerForm.GetCheckBool: Boolean;
begin
  Result := chkBool.Checked;
end;

procedure TCnIniFilerForm.SetCheckBool(const Value: Boolean);
begin
  chkBool.Checked := Value;
end;

procedure TCnIniFilerForm.chkIsAllStrClick(Sender: TObject);
begin
  chkBool.Enabled := chkIsAllStr.Checked;
end;

function TCnIniFilerForm.GetSectionMode: Boolean;
begin
  Result := chkSectionMode.Checked;
end;

procedure TCnIniFilerForm.SetSectionMode(const Value: Boolean);
begin
  chkSectionMode.Checked := Value;
end;

{$ENDIF CNWIZARDS_CNINIFILERWIZARD}
end.
