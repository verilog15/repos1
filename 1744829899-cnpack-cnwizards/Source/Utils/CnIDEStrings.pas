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

unit CnIDEStrings;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�IDE ��ص��ַ��������봦��Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin7 + Delphi 5.01
* ���ݲ��ԣ�
* �� �� �����ô����е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2024.08.01
*               ������Ԫ�����������������˴�
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  SysUtils, Classes, CnWideStrings;

type
{$IFDEF IDE_STRING_ANSI_UTF8}
  TCnIdeTokenString = WideString; // WideString for Utf8 Conversion
  PCnIdeTokenChar = PWideChar;
  TCnIdeTokenChar = WideChar;
  TCnIdeStringList = TCnWideStringList;
  TCnIdeTokenInt = Word;
{$ELSE}
  TCnIdeTokenString = string;     // Ansi/Utf16
  PCnIdeTokenChar = PChar;
  TCnIdeTokenChar = Char;
  TCnIdeStringList = TStringList;
  {$IFDEF UNICODE}
  TCnIdeTokenInt = Word;
  {$ELSE}
  TCnIdeTokenInt = Byte;
  {$ENDIF}
{$ENDIF}
  PCnIdeTokenInt = ^TCnIdeTokenInt;

function IDEWideCharIsWideLength(const AWChar: WideChar): Boolean; {$IFDEF SUPPORT_INLINE} inline; {$ENDIF}
{* �����ж�һ�� Unicode ���ַ��Ƿ�ռ�����ַ���ȣ���Ϊ������ IDE ����}

implementation

function IDEWideCharIsWideLength(const AWChar: WideChar): Boolean; {$IFDEF SUPPORT_INLINE} inline; {$ENDIF}
const
  CN_UTF16_ANSI_WIDE_CHAR_SEP = $1100;
var
  C: Integer;
begin
  C := Ord(AWChar);
  Result := C > CN_UTF16_ANSI_WIDE_CHAR_SEP; // ������Ϊ�� $1100 ��� Utf16 �ַ����ƿ�Ȳ�ռ���ֽ�
{$IFDEF DELPHI110_ALEXANDRIA_UP}
  if Result then // ����Щ��������ģ�������������ôд��
  begin
    if ((C >= $1470) and (C <= $14BF)) or
      ((C >= $16A0) and (C <= $16FF)) or
      ((C >= $1D00) and (C <= $1FFF)) or
      ((C >= $20A0) and (C <= $20BF)) or
      ((C >= $2550) and (C <= $256D)) or
      ((C >= $25A0) and (C <= $25BF) and (C <> $25B3) and (C <> $25BD)) then
      Result := False;
  end;
{$ENDIF}
end;

end.
