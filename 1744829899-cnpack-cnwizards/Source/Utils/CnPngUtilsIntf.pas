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

unit CnPngUtilsIntf;
{* |<PRE>
================================================================================
* ������ƣ�Cnpack IDE ר�Ұ���������
* ��Ԫ���ƣ�Png ��ʽ֧�ֵ�Ԫ
* ��Ԫ���ߣ��ܾ��� zjy@cnpack.org
* ��    ע������ pngimage �Ѿ��� Embarcadero �չ����µ����Э���ƺ������������Ŀ
*           ��Դ��Ϊ�˱����Ȩ���⣬�˴��� D2010 ��ʹ�ùٷ��� pngimage ����һ��
*           DLL �����Ͱ汾�� IDE ������ʹ�ã����� D10.4 �±��� 64 λ�汾��
* ����ƽ̨��Win7 + Delphi 2010/D10.4
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�ʹ����е��ַ����Ѿ����ػ�����ʽ
* �޸ļ�¼��2025.02.03 V1.1
*               ���� 64 λ��֧��
*           2011.07.05 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

type
  TCnConvertPngToBmpProc = function (PngFile, BmpFile: PAnsiChar): LongBool; stdcall;
  TCnConvertBmpToPngProc = function (BmpFile, PngFile: PAnsiChar): LongBool; stdcall;

function CnPngLibLoaded: LongBool;

function CnConvertPngToBmp(PngFile, BmpFile: string): LongBool; stdcall;

function CnConvertBmpToPng(BmpFile, PngFile: string): LongBool; stdcall;

implementation

uses
  Windows, SysUtils, CnCommon;

var
  FModuleHandle: HMODULE;
  FCnConvertPngToBmpProc: TCnConvertPngToBmpProc = nil;
  FCnConvertBmpToPngProc: TCnConvertBmpToPngProc = nil;

function ModulePath: string;
var
  ModName: array[0..MAX_PATH] of Char;
begin
  SetString(Result, ModName, GetModuleFileName(HInstance, ModName, SizeOf(ModName)));
  Result := _CnExtractFilePath(Result);
end;

procedure LoadCnPngLib;
var
  DllName: string;
begin
{$IFDEF WIN64}
  DllName := ModulePath + 'CnPngLib64.dll';
{$ELSE}
  DllName := ModulePath + 'CnPngLib.dll';
{$ENDIF}

  FModuleHandle := LoadLibrary(PChar(DllName));
  if FModuleHandle <> 0 then
  begin
    FCnConvertPngToBmpProc := TCnConvertPngToBmpProc(GetProcAddress(FModuleHandle, 'CnConvertPngToBmp'));
    FCnConvertBmpToPngProc := TCnConvertBmpToPngProc(GetProcAddress(FModuleHandle, 'CnConvertBmpToPng'));
  end;
end;

procedure FreeCnPngLib;
begin
  if FModuleHandle <> 0 then
  begin
    FreeLibrary(FModuleHandle);
    FCnConvertPngToBmpProc := nil;
    FCnConvertBmpToPngProc := nil;
    FModuleHandle := 0;
  end;
end;

function CnPngLibLoaded: LongBool;
begin
  Result := Assigned(FCnConvertPngToBmpProc) and Assigned(FCnConvertBmpToPngProc);
end;

function CnConvertPngToBmp(PngFile, BmpFile: string): LongBool; stdcall;
var
  P, B: AnsiString;
begin
  P := AnsiString(PngFile);
  B := AnsiString(BmpFile);
  if Assigned(FCnConvertPngToBmpProc) then
    Result := FCnConvertPngToBmpProc(PAnsiChar(P), PAnsiChar(B))
  else
    Result := False;
end;

function CnConvertBmpToPng(BmpFile, PngFile: string): LongBool; stdcall;
var
  P, B: AnsiString;
begin
  P := AnsiString(PngFile);
  B := AnsiString(BmpFile);
  if Assigned(FCnConvertBmpToPngProc) then
    Result := FCnConvertBmpToPngProc(PAnsiChar(B), PAnsiChar(P))
  else
    Result := False;
end;

initialization
  LoadCnPngLib;

finalization
  FreeCnPngLib;

end.
