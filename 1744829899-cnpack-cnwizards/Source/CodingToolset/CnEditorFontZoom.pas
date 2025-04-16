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

unit CnEditorFontZoom;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��༭���������ŵ�Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin XP SP3 + Delphi 5.01
* ���ݲ��ԣ�
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2010.06.10 V1.0
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNCODINGTOOLSETWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, IniFiles, ToolsAPI, CnWizClasses, CnWizUtils, CnConsts, CnCommon,
  Menus, CnCodingToolsetWizard, CnWizConsts, CnSelectionCodeTool;

type

//==============================================================================
// �༭����������
//==============================================================================

{ TCnEditorFontInc }

  TCnEditorFontInc = class(TCnSelectionCodeTool)
  public
    constructor Create(AOwner: TCnCodingToolsetWizard); override;
    function GetCaption: string; override;
    function GetHint: string; override;
    procedure GetToolsetInfo(var Name, Author, Email: string); override;
    procedure Execute; override;
  end;

//==============================================================================
// �༭��������С
//==============================================================================

{ TCnEditorFontDec }

  TCnEditorFontDec = class(TCnSelectionCodeTool)
  public
    constructor Create(AOwner: TCnCodingToolsetWizard); override;
    function GetCaption: string; override;
    function GetHint: string; override;
    procedure GetToolsetInfo(var Name, Author, Email: string); override;
    procedure Execute; override;
  end;

{$ENDIF CNWIZARDS_CNCODINGTOOLSETWIZARD}

implementation

{$IFDEF CNWIZARDS_CNCODINGTOOLSETWIZARD}

{ TCnEditorFontInc }

constructor TCnEditorFontInc.Create(AOwner: TCnCodingToolsetWizard);
begin
  inherited;
  ValidInSource := True;
  BlockMustNotEmpty := False;
end;

function TCnEditorFontInc.GetCaption: string;
begin
  Result := SCnEditorFontIncMenuCaption;
end;

function TCnEditorFontInc.GetHint: string;
begin
  Result := SCnEditorFontIncMenuHint;
end;

procedure TCnEditorFontInc.GetToolsetInfo(var Name, Author, Email: string);
begin
  Name := SCnEditorFontIncName;
  Author := SCnPack_Zjy;
  Email := SCnPack_ZjyEmail;
end;

procedure TCnEditorFontInc.Execute;
var
  Option: IOTAEditOptions;
begin
  Option := CnOtaGetEditOptions;
  if Assigned(Option) then
    Option.FontSize := Round(Option.FontSize * 1.1);
end;

{ TCnEditorFontDec }

constructor TCnEditorFontDec.Create(AOwner: TCnCodingToolsetWizard);
begin
  inherited;
  ValidInSource := True;
  BlockMustNotEmpty := False;
end;

function TCnEditorFontDec.GetCaption: string;
begin
  Result := SCnEditorFontDecMenuCaption;
end;

function TCnEditorFontDec.GetHint: string;
begin
  Result := SCnEditorFontDecMenuHint;
end;

procedure TCnEditorFontDec.GetToolsetInfo(var Name, Author, Email: string);
begin
  Name := SCnEditorFontDecName;
  Author := SCnPack_Zjy;
  Email := SCnPack_ZjyEmail;
end;

procedure TCnEditorFontDec.Execute;
var
  Option: IOTAEditOptions;
begin
  Option := CnOtaGetEditOptions;
  if Assigned(Option) then
    Option.FontSize := Round(Option.FontSize / 1.1);
end;

initialization
  RegisterCnCodingToolset(TCnEditorFontInc);
  RegisterCnCodingToolset(TCnEditorFontDec);

{$ENDIF CNWIZARDS_CNCODINGTOOLSETWIZARD}
end.
