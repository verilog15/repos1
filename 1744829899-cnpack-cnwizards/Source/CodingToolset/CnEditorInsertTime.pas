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

unit CnEditorInsertTime;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ���������ʱ�乤��
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWinXP SP2 + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2005.11.24 V1.0
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNCODINGTOOLSETWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, IniFiles, ToolsAPI, CnWizUtils, CnConsts, CnCommon, CnCodingToolsetWizard,
  CnWizConsts, CnSelectionCodeTool, CnIni, CnWizMultiLang;

type

//==============================================================================
// ������ɫ������
//==============================================================================

{ TCnEditorInsertTime }

  TCnEditorInsertTime = class(TCnBaseCodingToolset)
  private
    FDateTimeFmt: string;
  public
    constructor Create(AOwner: TCnCodingToolsetWizard); override;
    destructor Destroy; override;
    function GetCaption: string; override;
    function GetHint: string; override;
    procedure GetToolsetInfo(var Name, Author, Email: string); override;
    function GetState: TWizardState; override;
    procedure Execute; override;
  published
    property DateTimeFmt: string read FDateTimeFmt write FDateTimeFmt;
  end;

  TCnEditorInsertTimeForm = class(TCnTranslateForm)
    cbbDateTimeFmt: TComboBox;
    lblFmt: TLabel;
    lblPreview: TLabel;
    edtPreview: TEdit;
    btnOK: TButton;
    btnCancel: TButton;
    procedure cbbDateTimeFmtChange(Sender: TObject);
  private

  public
    procedure UpdateDateTimeStr;
  end;

{$ENDIF CNWIZARDS_CNCODINGTOOLSETWIZARD}

implementation

{$IFDEF CNWIZARDS_CNCODINGTOOLSETWIZARD}

{$R *.DFM}

{ TCnEditorInsertTime }

constructor TCnEditorInsertTime.Create(AOwner: TCnCodingToolsetWizard);
begin
  inherited;

end;

destructor TCnEditorInsertTime.Destroy;
begin

  inherited;
end;

function TCnEditorInsertTime.GetCaption: string;
begin
  Result := SCnEditorInsertTimeMenuCaption;
end;

function TCnEditorInsertTime.GetHint: string;
begin
  Result := SCnEditorInsertTimeMenuHint;
end;

procedure TCnEditorInsertTime.GetToolsetInfo(var Name, Author, Email: string);
begin
  Name := SCnEditorInsertTimeName;
  Author := SCnPack_LiuXiao;
  Email := SCnPack_LiuXiaoEmail;
end;

procedure TCnEditorInsertTime.Execute;
begin
  with TCnEditorInsertTimeForm.Create(nil) do
  begin
    if FDateTimeFmt = '' then
      cbbDateTimeFmt.ItemIndex := 0
    else
      cbbDateTimeFmt.Text := FDateTimeFmt;
    UpdateDateTimeStr;

    if ShowModal = mrOK then
    begin
      FDateTimeFmt := cbbDateTimeFmt.Text;
      CnOtaInsertTextToCurSource(edtPreview.Text, ipCur);
    end;
    Free;
  end;
end;

function TCnEditorInsertTime.GetState: TWizardState;
begin
  Result := inherited GetState;
  if (wsEnabled in Result) and not CurrentIsSource then
    Result := [];
end;

{ TCnInsertTimeForm }

procedure TCnEditorInsertTimeForm.UpdateDateTimeStr;
begin
  try
    edtPreview.Text := FormatDateTime(cbbDateTimeFmt.Text, Date + Time);
  except
    ;
  end;
end;

procedure TCnEditorInsertTimeForm.cbbDateTimeFmtChange(Sender: TObject);
begin
  UpdateDateTimeStr;
end;

initialization
  RegisterCnCodingToolset(TCnEditorInsertTime);
  
{$ENDIF CNWIZARDS_CNCODINGTOOLSETWIZARD}
end.
