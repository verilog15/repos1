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

unit CnZipWrapper;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ���Ŀ���ݹ��ܶ� CnZip ���ܵķ�װ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin7 + Delphi XE2/D10.4
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2018.08.26 V1.0 by liuxiao
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnPack.inc}

uses
  System.SysUtils,
  System.Classes,
  CnZip;

procedure CnWizStartZip(const SaveFileName: PAnsiChar; const Password: PAnsiChar;
  RemovePath: Boolean); stdcall;
{* ��ʼһ�� Zip�������ڲ�����ָ���ļ����������}

procedure CnWizZipAddFile(FileName, ArchiveFileName: PAnsiChar); stdcall;
{* ����ļ��� Zip������Ϊ��ʵ�ļ����Լ�Ҫд�� Zip �ļ����ļ���
  ��� ArchiveFileName �� nil����ʹ�� FileName ���� RemovePath ѡ�����}

procedure CnWizZipSetComment(Comment: PAnsiChar); stdcall;
{* ���� Zip �ļ�ע��}

function CnWizZipSaveAndClose: Boolean; stdcall;
{* ѹ������ Zip �ļ����ͷ��ڲ�����}

exports
  CnWizStartZip,
  CnWizZipAddFile,
  CnWizZipSetComment,
  CnWizZipSaveAndClose;

implementation

var
  FWriter: TCnZipWriter = nil;

procedure CnWizStartZip(const SaveFileName: PAnsiChar; const Password: PAnsiChar;
  RemovePath: Boolean); stdcall;
begin
  FreeAndNil(FWriter);
  FWriter := TCnZipWriter.Create;
  FWriter.RemovePath := RemovePath;
  FWriter.Password := Password;
  FWriter.CreateZipFile(SaveFileName);
end;

procedure CnWizZipAddFile(FileName, ArchiveFileName: PAnsiChar); stdcall;
begin
  if FWriter <> nil then
    FWriter.AddFile(FileName, ArchiveFileName);
end;

procedure CnWizZipSetComment(Comment: PAnsiChar); stdcall;
begin
  if FWriter <> nil then
    FWriter.Comment := Comment;
end;

function CnWizZipSaveAndClose: Boolean; stdcall;
begin
  Result := False;
  if FWriter <> nil then
  begin
    FWriter.Save;
    FWriter.Close;
    FreeAndNil(FWriter);
    Result := True;
  end;
end;

end.
