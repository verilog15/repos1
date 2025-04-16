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

unit CnPasCodeSample;
{* |<PRE>
================================================================================
* ������ƣ�CnPack ������Ԫ
* ��Ԫ���ƣ������Դ����ĵ����ɵ�ʾ����Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע���õ�ԪΪ���Դ����ĵ����ɵ�ʾ����Ӧ�и�������
* ����ƽ̨��PWin7 + Delphi 5
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2022.04.02 V1.0
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

{$I CnPack.inc}

interface

uses
  SysUtils, Classes;

const
  CN_DOC_TEST_MAX_LENGTH = 1024;
  {* ĳ���ͳ���ֵ������ĳ��󳤶�}

  CN_DOC_TEST_TITLE: string = 'Document Title';
  {* ĳ�ַ�������ֵ������ĳ����}

type

// =========================== ���ֲ�ͬ�������� ================================

  TCnDocTestObject = class;
  {* ĳ���ǰ������}

  TCnDocTestObject1 = class(TObject);
  {* ĳ�̳��࣬����ʽ���࣬��������}

  TCnDocTestObject2 = class
  {* ĳ�̳��࣬����ʽ����}
  end;

  TCnDocTestObject3 = class(TObject)
  {* ĳ�̳��࣬����ʽ����}
  end;

// =========================== ���ֲ�ͬ�Ľӿ����� ==============================

  ICnDocTestInterface1 = interface;
  {* ĳ�ӿڵ�ǰ������}

  ICnDocTestInterface2 = interface
  {* ĳ�̳нӿڣ�����ʽ����}
  end;

  ICnDocTestInterface3 = interface(IUnknown)
  {* ĳ�̳нӿڣ�����ʽ����}
  end;

// ================================ ������������ ===============================

  ECnDocTestException = class(Exception);
  {* ĳ�쳣}

  TCnDocTestEnum = (teNone, teTitle, teText);
  {* ĳö������}

  TCnDocTestEnums = set of TCnDocTestEnum;
  {* ĳ����ֵ}

  TCnDocTestArray = array[0..1023] of Cardinal;
  {* ĳ��������}

  PCnDocTestArray = ^TCnDocTestArray;
  {* ĳ����ָ��}

  TCnDocTestRecord = packed record
  {* ĳ���սṹ}
    RecordField1: Cardinal;
    {* ĳ�ṹ��Աһ}
    RecordField2: Boolean;
    {* ĳ�ṹ��Ա��}
  end;

  ICnDocTestInterface = interface(IUnknown)
  {* ĳ�̳нӿڣ���ͷ�г�Ա����������}
  ['{D735C546-338D-4108-ABF8-3A2D23D93FD1}']
    procedure TestInterfaceProcedure;
    {* ĳ�ӿڵ�ĳ����}
    function GetTestProperty: Boolean;
    {* ĳ�ӿڵ�ĳ���Ժ���}
    property TestProperty: Boolean read GetTestProperty;
    {* ĳ�ӿڵ�ĳ����}
  end;

  TCnDocTestObject = class(TObject)
  {* ĳ���Դ��࣬�и�������}
  private
    FAProperty: Integer;
    function GetItem(Index: Integer): string;
    procedure SetAProperty(const Value: Integer);
    procedure SetItem(Index: Integer; const Value: string);

  protected
    FTestField: Boolean;
    {* ĳ�����ֶ�}
    procedure DoTest; virtual;
    {* ĳ�������ⷽ��}
  public
    constructor Create(AOwner: TObject); virtual;
    {* ���캯��}
    destructor Destroy; override;
    {* ��������}
  private
    procedure Init;
    {* ��ʼ������}
    function IsReady: Boolean;
    {* �����Ƿ�׼����}
  protected
    property Items[Index: Integer]: string read GetItem write SetItem; default;
    {* ĳ��������}

  published
    property AProperty: Integer read FAProperty write SetAProperty;
    {* ĳ����}
  end;

  ICnDocTestInterface1 = interface
  {* ĳû�̳нӿڣ���ͷ�г�Ա����}
    procedure TestInterfaceProcedure1;
    {* ĳ�ӿڵ�ĳ����}
  end;

  TCnDocTestObject4 = class
  {* ĳ��һ������}
    FTestField1: Integer;
    {* ĳ�޷�Χ�� Field1}
    FTesetField2: string;
    {* ĳ�޷�Χ�� Field2}
  public
    FTestField3: Cardinal;
    {* ĳ�з�Χ�� Field3}
  end;

procedure CnDocTestProcedure(Param1: Integer; var Param2: string); stdcall;
{* ĳȫ�ֹ��̣��� Directive}

function CnDocTestFunction(const Param1: Integer; Param2: string = ''): Integer;
{* ĳȫ�ֺ���}

var
  CnDocTestGlobalVar1: TCnDocTestEnum = teNone;
  {* ĳȫ�ֱ���һ}

  CnDocTestGlobalVar2: TCnDocTestArray;
  {* ĳȫ�ֱ�����}

implementation

procedure CnDocTestProcedure(Param1: Integer; var Param2: string); stdcall;
begin

end;

function CnDocTestFunction(const Param1: Integer; Param2: string = ''): Integer;
begin
  Result := 0;
end;

{ TCnDocTestObject }

constructor TCnDocTestObject.Create(AOwner: TObject);
begin
  if not IsReady then
    Init;
end;

destructor TCnDocTestObject.Destroy;
begin
  inherited;

end;

procedure TCnDocTestObject.DoTest;
begin

end;

function TCnDocTestObject.GetItem(Index: Integer): string;
begin

end;

procedure TCnDocTestObject.Init;
begin

end;

function TCnDocTestObject.IsReady: Boolean;
begin
  Result := False;
end;

procedure TCnDocTestObject.SetAProperty(const Value: Integer);
begin
  FAProperty := Value;
end;

procedure TCnDocTestObject.SetItem(Index: Integer; const Value: string);
begin

end;

end.
