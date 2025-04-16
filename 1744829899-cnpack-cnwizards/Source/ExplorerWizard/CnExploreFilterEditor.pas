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

unit CnExploreFilterEditor;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��ļ�������ר�ҵ�Ԫ
* ��Ԫ���ߣ�Hhha��Hhha�� Hhha@eyou.con
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 7
* ���ݲ��ԣ�δ����
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNEXPLORERWIZARD}

uses
  Windows, Forms, Classes, Controls, StdCtrls, CnWizMultiLang;

type
  TCnExploreFilterEditorForm = class(TCnTranslateForm)
    OKBtn: TButton;
    CancelBtn: TButton;
    lbl1: TLabel;
    lbl2: TLabel;
    edtType: TEdit;
    edtExtName: TEdit;
  private

  public

  end;

var
  CnExploreFilterEditorForm: TCnExploreFilterEditorForm;

{$ENDIF CNWIZARDS_CNEXPLORERWIZARD}

implementation

{$IFDEF CNWIZARDS_CNEXPLORERWIZARD}

{$R *.DFM}

{$ENDIF CNWIZARDS_CNEXPLORERWIZARD}
end.
