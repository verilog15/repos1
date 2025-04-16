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

unit CnAICoderNetClient;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�AI ��������ר�ҵ������߳����ݶ���Ԫ
* ��Ԫ���ߣ�CnPack ������
* ��    ע��
* ����ƽ̨��PWin7 + Delphi 5.01
* ���ݲ��ԣ�PWin7/10/11 + Delphi/C++Builder
* �� �� �����ô����е��ַ����ݲ�֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2024.05.01 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNAICODERWIZARD}

uses
  SysUtils, Classes, CnNative, CnThreadPool;

type
  TCnAIRequestType = (artRaw, artExplainCode, artReviewCode, artGenTestCase);
  {* ��������}

  TCnAINetRequestDataObject = class;

  TCnAIAnswerCallback = procedure(StreamMode, Partly, Success, IsStreamEnd: Boolean;
    SendId: Integer; const Answer: string; ErrorCode: Cardinal; Tag: TObject) of object;
  {* ���� AI �󷵻صĽ���ص��¼���Success ��ʾ�ɹ��������ɹ���Answer ��ʾ�ظ�������
    Partly Ϊ True ��ʾ�����Ƕ�η����е�һ�Σ�Tag �Ƿ�������ʱ����� Tag}

  TCnAINetDataResponse = procedure(Success, Partly: Boolean; Thread: TCnPoolingThread;
    DataObj: TCnAINetRequestDataObject; Data: TBytes; ErrCode: Cardinal) of object;
  {* ��������Ļص������߳ɹ���񣬳ɹ��� Data ��������
    Partly Ϊ True ��ʾ�����Ƕ�η����е�һ��}

  TCnAINetRequestThread = class(TCnPoolingThread)
  {* �̳߳��е��߳�ʵ��}
  private
    FData: TBytes;
    FSendId: Integer;
  public
    property SendId: Integer read FSendId write FSendId;
    property Data: TBytes read FData write FData;
  end;

  TCnAINetRequestDataObject = class(TCnTaskDataObject)
  {* ������������������࣬�ɷ����߸�����������������������Ӹ��̳߳�
    �н��ʱ�̻߳�ص� OnResponse �¼�}
  private
    FURL: string;
    FSendId: Integer;
    FData: TBytes;
    FStreamMode: Boolean;
    FTag: TObject;
    FOnResponse: TCnAINetDataResponse;
    FRequestType: TCnAIRequestType;
    FOnAnswer: TCnAIAnswerCallback;
  public
    function Clone: TCnTaskDataObject; override;

    property StreamMode: Boolean read FStreamMode write FStreamMode;
    {* �����Ƿ��� Stream ģʽ���ɷ���������·����ݶ�����һ����ȫ����}

    property RequestType: TCnAIRequestType read FRequestType write FRequestType;
    {* ��������}
    property Data: TBytes read FData write FData;
    {* ������װ�õ��������ݣ���ʵ����ҵ���߼����������� HTTP ͷ����֤����}

    property SendId: Integer read FSendId write FSendId;
    {* ���� ID ����}
    property URL: string read FURL write FURL;
    {* �����ַ}
    property Tag: TObject read FTag write FTag;
    {* ���õ�һ�� Object ���ã��������������յ���Ӧʱ���ݹ���}

    property OnAnswer: TCnAIAnswerCallback read FOnAnswer write FOnAnswer;
    {* �������ߵĻص��¼���һ�����û����õ��û�������}
    property OnResponse: TCnAINetDataResponse read FOnResponse write FOnResponse;
    {* �յ���������ʱ�Ļص��¼���һ������������õ������ڲ�
      ע���������߳��б����õģ�����ʱ���� Synchronize �����߳����輰ʱ��������}
  end;

{$ENDIF CNWIZARDS_CNAICODERWIZARD}

implementation

{$IFDEF CNWIZARDS_CNAICODERWIZARD}

{ TCnAINetRequestDataObject }

function TCnAINetRequestDataObject.Clone: TCnTaskDataObject;
begin
  Result := TCnAINetRequestDataObject.Create;

  // ע����� TCnAINetRequestDataObject �����������ԣ��˴�Ҫͬ������
  TCnAINetRequestDataObject(Result).Tag := FTag;
  TCnAINetRequestDataObject(Result).URL := FURL;
  TCnAINetRequestDataObject(Result).SendId := FSendId;
  TCnAINetRequestDataObject(Result).RequestType := FRequestType;
  TCnAINetRequestDataObject(Result).StreamMode := FStreamMode;
  TCnAINetRequestDataObject(Result).OnResponse := FOnResponse;
  TCnAINetRequestDataObject(Result).OnAnswer := FOnAnswer;
end;

{$ENDIF CNWIZARDS_CNAICODERWIZARD}
end.
