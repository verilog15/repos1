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

unit CnWizBoot;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�ר����������
* ��Ԫ���ߣ����壨QSoft��  qsoft@cnpack.org
* ��    ע�������û�����������Delphiʱ������Shift�������ù��ߣ�������ʱ����/����
*           ר�ҡ�
*
* ����ƽ̨��PWin2000Pro + Delphi 5.62
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2003.10.03 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  CnWizMultiLang, ComCtrls, StdCtrls, Buttons, ToolWin, CnWizConsts, CnWizOptions;

type
  TCnWizBootForm = class(TCnTranslateForm)
    lvWizardsList: TListView;
    ToolBar1: TToolBar;
    tbnSelectAll: TToolButton;
    tbnUnSelect: TToolButton;
    tbnReverseSelect: TToolButton;
    tbtnOK: TToolButton;
    ToolButton5: TToolButton;
    tbtnCancel: TToolButton;
    stbStatusbar: TStatusBar;
    procedure FormShow(Sender: TObject);
    procedure tbtnOKClick(Sender: TObject);
    procedure tbtnCancelClick(Sender: TObject);
    procedure tbnSelectAllClick(Sender: TObject);
    procedure tbnUnSelectClick(Sender: TObject);
    procedure tbnReverseSelectClick(Sender: TObject);
    procedure lvWizardsListClick(Sender: TObject);
    procedure FormKeyPress(Sender: TObject; var Key: Char);
  private
    procedure UpdateStatusBar;
  public
    procedure GetBootList(var ABoots: array of boolean);
  end;

implementation

uses CnWizClasses, CnWizManager;

{$R *.DFM}

procedure TCnWizBootForm.FormShow(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to GetCnWizardClassCount - 1 do
  begin
    with lvWizardsList.Items.Add do
    begin
      Caption := IntToStr(I + 1);
      SubItems.Add(TCnWizardClass(GetCnWizardClassByIndex(I)).WizardName);
      SubItems.Add(TCnWizardClass(GetCnWizardClassByIndex(I)).GetIDStr);
      SubItems.Add(GetCnWizardTypeNameFromClass(TCnWizardClass(GetCnWizardClassByIndex(I))));

      Checked := WizOptions.ReadBool(SCnBootLoadSection,
        TCnWizardClass(GetCnWizardClassByIndex(I)).ClassName,
        CnWizardMgr.WizardCanCreate[TCnWizardClass(GetCnWizardClassByIndex(I)).ClassName]);
    end;
  end;
  UpdateStatusBar;
end;

procedure TCnWizBootForm.UpdateStatusBar;
var
  I, Count: Integer;
begin
  Count := 0;
  for I := 0 to lvWizardsList.Items.Count - 1 do
  begin
    if lvWizardsList.Items[I].Checked then
      Inc(Count);
  end;
  
  stbStatusbar.Panels[1].Text := Format(SCnWizBootCurrentCount, [lvWizardsList.Items.Count]);
  stbStatusbar.Panels[2].Text := Format(SCnWizBootEnabledCount, [Count]);
end;

procedure TCnWizBootForm.GetBootList(var ABoots: array of boolean);
var
  I: Integer;
begin
  for I := 0 to lvWizardsList.Items.Count - 1 do
    ABoots[I] := lvWizardsList.Items[I].Checked;
end;

procedure TCnWizBootForm.tbtnOKClick(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to GetCnWizardClassCount - 1 do
  begin
    WizOptions.WriteBool(SCnBootLoadSection,
      TCnWizardClass(GetCnWizardClassByIndex(I)).ClassName,
      lvWizardsList.Items[I].Checked);
  end;
  ModalResult := mrOK;
end;

procedure TCnWizBootForm.tbtnCancelClick(Sender: TObject);
begin
  ModalResult := mrCancel;
end;

procedure TCnWizBootForm.tbnSelectAllClick(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to lvWizardsList.Items.Count - 1 do
    lvWizardsList.Items[I].Checked := True;

  UpdateStatusBar;
end;

procedure TCnWizBootForm.tbnUnSelectClick(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to lvWizardsList.Items.Count - 1 do
    lvWizardsList.Items[I].Checked := False;

  UpdateStatusBar;
end;

procedure TCnWizBootForm.tbnReverseSelectClick(Sender: TObject);
var
  I: Integer;
begin
  for I := 0 to lvWizardsList.Items.Count - 1 do
    lvWizardsList.Items[I].Checked := not lvWizardsList.Items[I].Checked;

  UpdateStatusBar;
end;

procedure TCnWizBootForm.lvWizardsListClick(Sender: TObject);
begin
  UpdateStatusBar;
end;

procedure TCnWizBootForm.FormKeyPress(Sender: TObject; var Key: Char);
begin
  if Key = #27 then
    tbtnCancelClick(nil);
end;

end.

