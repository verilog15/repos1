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

unit CnPropertyCompConfigFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnPack ר�Ұ�
* ��Ԫ���ƣ�������ԶԱ����ô��嵥Ԫ
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
* ��    ע��
* ����ƽ̨��Win7 + Delphi 5
* ���ݲ��ԣ�δ����
* �� �� �����ô����е��ַ����ݲ����ϱ��ػ�����ʽ
* �޸ļ�¼��2021.08.08
*               ������Ԫ��ʵ�ֻ�������
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNDESIGNWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ComCtrls, CnWizMultiLang, ExtCtrls;

type
  TCnPropertyCompConfigForm = class(TCnTranslateForm)
    btnOK: TButton;
    btnCancel: TButton;
    btnHelp: TButton;
    pgc1: TPageControl;
    tsProperty: TTabSheet;
    chkSameType: TCheckBox;
    lblAll: TLabel;
    mmoIgnoreProperties: TMemo;
    chkShowMenu: TCheckBox;
    tsFont: TTabSheet;
    pnlFont: TPanel;
    btnFont: TButton;
    dlgFont: TFontDialog;
    btnReset: TButton;
    procedure btnHelpClick(Sender: TObject);
    procedure btnFontClick(Sender: TObject);
    procedure btnResetClick(Sender: TObject);
  private
    FFontChanged: Boolean;
  protected
    function GetHelpTopic: string; override;
  public
    property FontChanged: Boolean read FFontChanged;
  end;

var
  CnPropertyCompConfigForm: TCnPropertyCompConfigForm;

{$ENDIF CNWIZARDS_CNDESIGNWIZARD}

implementation

{$IFDEF CNWIZARDS_CNDESIGNWIZARD}

{$R *.DFM}

uses
  CnGraphUtils;

{ TCnPropertyCompConfigForm }

function TCnPropertyCompConfigForm.GetHelpTopic: string;
begin
  Result := 'CnAlignSizeConfig';
end;

procedure TCnPropertyCompConfigForm.btnHelpClick(Sender: TObject);
begin
  ShowFormHelp;
end;

procedure TCnPropertyCompConfigForm.btnFontClick(Sender: TObject);
begin
  dlgFont.Font := pnlFont.Font;
  if dlgFont.Execute then
  begin
    pnlFont.Font := dlgFont.Font;
    FFontChanged := True;
  end;
end;

procedure TCnPropertyCompConfigForm.btnResetClick(Sender: TObject);
var
  OldFont: TFont;
begin
  OldFont := TFont.Create;
  try
    OldFont.Assign(pnlFont.Font);
    pnlFont.ParentFont := True;

    // ���ú�������б仯����
    FFontChanged := not FontEqual(OldFont, pnlFont.Font);
  finally
    OldFont.Free;
  end;
end;

{$ENDIF CNWIZARDS_CNDESIGNWIZARD}
end.
