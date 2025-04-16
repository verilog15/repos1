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

unit CnVclToFmxImpl;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�VCL �� FMX ת�����ܵĽӿ�ʵ��
* ��Ԫ���ߣ�CnPack ������
* ��    ע���õ�Ԫ�� VCL �� FMX ת�����ܵĽӿ�ʵ��
* ����ƽ̨��Win7 + Delphi 5.0
* ���ݲ��ԣ�����ƽ̨
* �� �� ��������Ҫ
* �޸ļ�¼��2022.05.11 V1.0
*               ������Ԫ��
================================================================================
|</PRE>}

interface

{$I CnPack.inc}

uses
  System.SysUtils, System.Classes, CnVclToFmxIntf;

type
  TCnVclToFmxImpl = class(TInterfacedObject, ICnVclToFmxIntf)
  private
    FFmx: string;
    FPas: string;
  public
    function OpenAndConvertFile(InDfmFile: PWideChar): Boolean;
    function SaveNewFile(InNewFile: PWideChar): Boolean;
  end;

function GetVclToFmxConverter: ICnVclToFmxIntf; stdcall;

exports
  GetVclToFmxConverter;

implementation

uses
  CnVclToFmxConverter;

var
  FImpl: ICnVclToFmxIntf = nil;

function GetVclToFmxConverter: ICnVclToFmxIntf;
begin
  if FImpl = nil then
    FImpl := TCnVclToFmxImpl.Create;
  Result := FImpl;
end;

{ TCnVclToFmxImpl }

function TCnVclToFmxImpl.OpenAndConvertFile(InDfmFile: PWideChar): Boolean;
begin
  FFmx := '';
  FPas := '';
  Result := CnVclToFmxConvert(InDfmFile, FFmx, FPas);
end;

function TCnVclToFmxImpl.SaveNewFile(InNewFile: PWideChar): Boolean;
var
  F: string;
begin
  if (FFmx <> '') and (InNewFile <> nil) then
  begin
    F := InNewFile;
    Result := CnVclToFmxSaveContent(ChangeFileExt(F, '.fmx'), FFmx);
    if Result and (FPas <> '') then
    begin
      FPas := CnVclToFmxReplaceUnitName(F, FPas);
      Result := CnVclToFmxSaveContent(ChangeFileExt(F, '.pas'), FPas);
    end;
  end;
end;

initialization

finalization
  FImpl := nil;

end.
