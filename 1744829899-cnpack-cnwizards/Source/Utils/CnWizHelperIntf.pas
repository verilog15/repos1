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

unit CnWizHelperIntf;

interface
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�CnWizHelper.dll �Ľӿ�
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2010.05.1 V1.0 by zjy
*               ������Ԫ
================================================================================
|</PRE>}

uses
  Windows, SysUtils, CnCommon;

const
{$IFDEF WIN64}
  SCnWizHelperDllName = 'CnWizHelper64.Dll';
  SCnWizZipDllName = 'CnZipUtils64.Dll';
{$ELSE}
  SCnWizHelperDllName = 'CnWizHelper.Dll';
  SCnWizZipDllName = 'CnZipUtils.Dll';
{$ENDIF}

type
  TProcCnWizStartZip = procedure(const SaveFileName: PAnsiChar; const Password: PAnsiChar;
    RemovePath: Boolean); stdcall;
  {* ��ʼһ�� Zip�������ڲ�����ָ���ļ����������}

  TProcCnWizZipAddFile = procedure(FileName, ArchiveFileName: PAnsiChar); stdcall;
  {* ����ļ��� Zip������Ϊ��ʵ�ļ����Լ�Ҫд�� Zip �ļ����ļ���
    ��� ArchiveFileName �� nil����ʹ�� FileName ���� RemovePath ѡ�����}

  TProcCnWizZipSetComment = procedure(Comment: PAnsiChar); stdcall;
  {* ���� Zip �ļ�ע��}

  TFuncCnWizZipSaveAndClose = function: Boolean; stdcall;
  {* ѹ������ Zip �ļ����ͷ��ڲ�����}

  TFuncCnWizInetGetFile = function(AURL, FileName: PAnsiChar): Boolean; stdcall;
  {* ͨ�����������ȡ URL ���ݲ����浽�ļ�}

function CnWizHelperLoaded: Boolean;

function CnWizZipUtilsLoaded: Boolean;

//------------------------------------------------------------------------------
// ZIP ����
//------------------------------------------------------------------------------

function CnWizHelperZipValid: Boolean;

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

//------------------------------------------------------------------------------
// InetUtils ����
//------------------------------------------------------------------------------

function CnWizHelperInetValid: Boolean;

function CnWiz_Inet_GetFile(AURL, FileName: PAnsiChar): Boolean; stdcall;

implementation

{$IFDEF DEBUG}
uses
  CnDebug;
{$ENDIF}

var
  HelperDllHandle: HMODULE = 0;
  ZipDllHandle: HMODULE = 0;

  FCnWizStartZip: TProcCnWizStartZip;
  FCnWizZipAddFile: TProcCnWizZipAddFile;
  FCnWizZipSetComment: TProcCnWizZipSetComment;
  FCnWizZipSaveAndClose: TFuncCnWizZipSaveAndClose;
  FCnWizInetGetFile: TFuncCnWizInetGetFile;

procedure LoadWizHelperDll;
var
  ModuleName: array[0..MAX_Path - 1] of Char;
begin
  GetModuleFileName(hInstance, ModuleName, MAX_PATH);
  HelperDllHandle := LoadLibrary(PChar(_CnExtractFilePath(ModuleName) + SCnWizHelperDllName));
  ZipDllHandle := LoadLibrary(PChar(_CnExtractFilePath(ModuleName) + SCnWizZipDllName));
  
  if HelperDllHandle <> 0 then
  begin
    FCnWizInetGetFile := TFuncCnWizInetGetFile(GetProcAddress(HelperDllHandle, 'CnWiz_Inet_GetFile'));
  end
  else
  begin
{$IFDEF DEBUG}
    CnDebugger.LogMsg('Load CnWizHelper.dll failed.');
{$ENDIF}
  end;

  if ZipDllHandle <> 0 then
  begin
    FCnWizStartZip := TProcCnWizStartZip(GetProcAddress(ZipDllHandle, 'CnWizStartZip'));
    FCnWizZipAddFile := TProcCnWizZipAddFile(GetProcAddress(ZipDllHandle, 'CnWizZipAddFile'));
    FCnWizZipSetComment := TProcCnWizZipSetComment(GetProcAddress(ZipDllHandle, 'CnWizZipSetComment'));
    FCnWizZipSaveAndClose := TFuncCnWizZipSaveAndClose(GetProcAddress(ZipDllHandle, 'CnWizZipSaveAndClose'));
  end
  else
  begin
{$IFDEF DEBUG}
    CnDebugger.LogMsg('Load CnZipUtils.dll failed.');
{$ENDIF}
  end;

{$IFDEF DEBUG}
  CnDebugger.LogBoolean(CnWizHelperZipValid, 'CnWizHelperZipValid');
  CnDebugger.LogBoolean(CnWizHelperInetValid, 'CnWizHelperInetValid');
{$ENDIF}
end;

procedure FreeWizHelperDll;
begin
  if HelperDllHandle <> 0 then
  begin
    FreeLibrary(HelperDllHandle);
    HelperDllHandle := 0;
  end;

  if ZipDllHandle <> 0 then
  begin
    FreeLibrary(ZipDllHandle);
    ZipDllHandle := 0;
  end;
end;  

function CnWizHelperLoaded: Boolean;
begin
  Result := HelperDllHandle <> 0;
end;

function CnWizZipUtilsLoaded: Boolean;
begin
  Result := ZipDllHandle <> 0;
end;

//------------------------------------------------------------------------------
// ZIP ����
//------------------------------------------------------------------------------

function CnWizHelperZipValid: Boolean;
begin
  Result := CnWizZipUtilsLoaded and Assigned(FCnWizStartZip) and
    Assigned(FCnWizZipAddFile) and Assigned(FCnWizZipSetComment)
    and Assigned(FCnWizZipSaveAndClose);
end;  

procedure CnWizStartZip(const SaveFileName: PAnsiChar; const Password: PAnsiChar;
  RemovePath: Boolean); stdcall;
begin
  if CnWizHelperZipValid then
    FCnWizStartZip(SaveFileName, Password, RemovePath);
end;  

procedure CnWizZipAddFile(FileName, ArchiveFileName: PAnsiChar); stdcall;
begin
  if CnWizHelperZipValid then
    FCnWizZipAddFile(FileName, ArchiveFileName);
end;

procedure CnWizZipSetComment(Comment: PAnsiChar); stdcall;
begin
  if CnWizHelperZipValid then
    FCnWizZipSetComment(Comment);
end;

function CnWizZipSaveAndClose: Boolean; stdcall;
begin
  if CnWizHelperZipValid then
    Result := FCnWizZipSaveAndClose
  else
    Result := False;
end;

//------------------------------------------------------------------------------
// InetUtils ����
//------------------------------------------------------------------------------

function CnWizHelperInetValid: Boolean;
begin
  Result := CnWizHelperLoaded and Assigned(FCnWizInetGetFile);
end;

function CnWiz_Inet_GetFile(AURL, FileName: PAnsiChar): Boolean; stdcall;
begin
  if CnWizHelperInetValid then
    Result := FCnWizInetGetFile(AURL, FileName)
  else
    Result := False;
end;  

initialization
  LoadWizHelperDll;

finalization
  FreeWizHelperDll;

end.
