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

unit CnAICoderEngineImpl;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�AI ��������ר�ҵ�����ʵ�ֵ�Ԫ
* ��Ԫ���ߣ�CnPack ������
* ��    ע���� Claude/Gemini ������̥��֧�� OpenAI ���ݸ�ʽ
* ����ƽ̨��PWin7 + Delphi 5.01
* ���ݲ��ԣ�PWin7/10/11 + Delphi/C++Builder
* �� �� �����ô����е��ַ����ݲ�֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2024.05.04 V1.0
*               ͨ��ǧ�ʼ��ݻ��� API �ӿڸ�ʽ
*           2024.05.04 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNAICODERWIZARD}

uses
  SysUtils, Classes, Contnrs, CnNative, CnJSON, CnAICoderEngine,
  CnAICoderNetClient, CnAICoderConfig;

type
  TCnOpenAIAIEngine = class(TCnAIBaseEngine)
  {* OpenAI ����}
  public
    class function EngineName: string; override;
  end;

  TCnMistralAIAIEngine = class(TCnAIBaseEngine)
  {* MistralAI ����}
  public
    class function EngineName: string; override;
  end;

  TCnClaudeAIEngine = class(TCnAIBaseEngine)
  {* Claude ����}
  protected
    // Claude �������֤ͷ��Ϣ������������ͬ
    procedure PrepareRequestHeader(Headers: TStringList); override;

    // Claude �� HTTP �ӿڵ� JSON ��ʽ����������������ͬ
    function ConstructRequest(RequestType: TCnAIRequestType; const Code: string): TBytes; override;

    // Claude ����Ϣ���ظ�ʽҲ��ͬ
    function ParseResponse(SendId: Integer; StreamMode, Partly: Boolean; var Success: Boolean;
      var ErrorCode: Cardinal; var IsStreamEnd: Boolean; RequestType: TCnAIRequestType;
      const Response: TBytes): string; override;
  public
    class function EngineName: string; override;
    class function OptionClass: TCnAIEngineOptionClass; override;
  end;

  TCnGeminiAIEngine = class(TCnAIBaseEngine)
  {* Gemini ����}
  protected
    // Gemini �� URL ������������ͬ
    function GetRequestURL(DataObj: TCnAINetRequestDataObject): string; override;

    // Gemini �������֤ͷ��Ϣ������������ͬ
    procedure PrepareRequestHeader(Headers: TStringList); override;

    // Gemini �� HTTP �ӿڵ� JSON ��ʽ����������������ͬ
    function ConstructRequest(RequestType: TCnAIRequestType; const Code: string): TBytes; override;

    // Gemini ����Ϣ���ظ�ʽҲ��ͬ
    function ParseResponse(SendId: Integer; StreamMode, Partly: Boolean; var Success: Boolean;
      var ErrorCode: Cardinal; var IsStreamEnd: Boolean; RequestType: TCnAIRequestType;
      const Response: TBytes): string; override;
  public
    class function EngineName: string; override;
  end;

  TCnQWenAIEngine = class(TCnAIBaseEngine)
  {* ͨ��ǧ�� AI ����}
  public
    class function EngineName: string; override;
  end;

  TCnMoonshotAIEngine = class(TCnAIBaseEngine)
  {* ��֮���� AI ����}
  public
    class function EngineName: string; override;
  end;

  TCnChatGLMAIEngine = class(TCnAIBaseEngine)
  {* �������� AI ����}
  public
    class function EngineName: string; override;
  end;

  TCnBaiChuanAIEngine = class(TCnAIBaseEngine)
  {* �ٴ����� AI ����}
  public
    class function EngineName: string; override;
  end;

  TCnDeepSeekAIEngine = class(TCnAIBaseEngine)
  {* DeepSeek�����������AI ����}
  public
    class function EngineName: string; override;
  end;

  TCnOllamaAIEngine = class(TCnAIBaseEngine)
  {* ���ػ�˽�л�����ļ��� Ollama ����}
  protected
    function ConstructRequest(RequestType: TCnAIRequestType; const Code: string): TBytes; override;
    function ParseResponse(SendId: Integer; StreamMode, Partly: Boolean; var Success: Boolean;
      var ErrorCode: Cardinal; var IsStreamEnd: Boolean; RequestType: TCnAIRequestType;
      const Response: TBytes): string; override;
  public
    class function EngineName: string; override;
    class function NeedApiKey: Boolean; override;
  end;

{$ENDIF CNWIZARDS_CNAICODERWIZARD}

implementation

{$IFDEF CNWIZARDS_CNAICODERWIZARD}

const
  CRLF = #13#10;
  LF = #10;

{ TCnOpenAIEngine }

class function TCnOpenAIAIEngine.EngineName: string;
begin
  Result := 'OpenAI';
end;

{ TCnQWenAIEngine }

class function TCnQWenAIEngine.EngineName: string;
begin
  Result := 'ͨ��ǧ��';
end;

{ TCnMoonshotAIEngine }

class function TCnMoonshotAIEngine.EngineName: string;
begin
  Result := '��֮����';
end;

{ TCnChatGLMAIEngine }

class function TCnChatGLMAIEngine.EngineName: string;
begin
  Result := '��������';
end;

{ TCnBaiChuanAIEngine }

class function TCnBaiChuanAIEngine.EngineName: string;
begin
  Result := '�ٴ�����';
end;

{ TCnDeepSeekAIEngine }

class function TCnDeepSeekAIEngine.EngineName: string;
begin
  Result := 'DeepSeek';
end;

{ TCnMistralAIAIEngine }

class function TCnMistralAIAIEngine.EngineName: string;
begin
  Result := 'MistralAI';
end;

{ TCnClaudeAIEngine }

function TCnClaudeAIEngine.ConstructRequest(RequestType: TCnAIRequestType;
  const Code: string): TBytes;
var
  ReqRoot, Msg: TCnJSONObject;
  Arr: TCnJSONArray;
  S: AnsiString;
begin
  ReqRoot := TCnJSONObject.Create;
  try
    ReqRoot.AddPair('model', Option.Model);
    ReqRoot.AddPair('stream', Option.Stream);
    ReqRoot.AddPair('temperature', Option.Temperature);
    ReqRoot.AddPair('max_tokens', (Option as TCnClaudeAIEngineOption).MaxTokens);

    ReqRoot.AddPair('system', Option.SystemMessage); // Claude ����� System ��Ϣ
    Arr := ReqRoot.AddArray('messages');

    Msg := TCnJSONObject.Create;
    Msg.AddPair('role', 'user');
    if RequestType = artExplainCode then
      Msg.AddPair('content', Option.ExplainCodePrompt + #13#10 + Code)
    else if RequestType = artReviewCode then
      Msg.AddPair('content', Option.ReviewCodePrompt + #13#10 + Code)
    else if RequestType = artGenTestCase then
      Msg.AddPair('content', Option.GenTestCasePrompt + #13#10 + Code)
    else if RequestType = artRaw then
      Msg.AddPair('content', Code);

    Arr.AddValue(Msg);

    S := ReqRoot.ToJSON;
    Result := AnsiToBytes(S);
  finally
    ReqRoot.Free;
  end;
end;

class function TCnClaudeAIEngine.EngineName: string;
begin
  Result := 'Claude';
end;

class function TCnClaudeAIEngine.OptionClass: TCnAIEngineOptionClass;
begin
  Result := TCnClaudeAIEngineOption;
end;

function TCnClaudeAIEngine.ParseResponse(SendId: Integer; StreamMode, Partly: Boolean;
  var Success: Boolean; var ErrorCode: Cardinal; var IsStreamEnd: Boolean;
  RequestType: TCnAIRequestType; const Response: TBytes): string;
var
  RespRoot, Msg: TCnJSONObject;
  Arr: TCnJSONArray;
  S, Prev: AnsiString;
  PrevS: string;
  P: PAnsiChar;
  HasPartly: Boolean;
  JsonObjs: TObjectList;
  I, Step: Integer;
begin
// ����ʽ��ʡ���˸��ӵģ�
// data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
// data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
// data: {"type": "message_stop"}

  Result := '';
  // ���� SendId �ұ��λỰ�����������
  Prev := '';
  if FPrevRespRemainMap.Find(IntToStr(SendId), PrevS) then
  begin
    Prev := AnsiString(PrevS);
    S := Prev + BytesToAnsi(Response) // ��ʣ������ƴ�����������ٴν��н���
  end
  else
    S := BytesToAnsi(Response);

  JsonObjs := TObjectList.Create(True);
  Step := CnJSONParse(PAnsiChar(S), JsonObjs);
  P := PAnsiChar(PAnsiChar(S) + Step);

  if (P <> #0) and (Step < Length(S)) then
  begin
    // ����û�����꣨����û���㣩���Ҳ��ǽ�������˵����ʣ������
    Prev := Copy(S, Step + 1, MaxInt);
  end
  else // ˵��ûʣ������
    Prev := '';

  // ����ʣ�඼������
  FPrevRespRemainMap.Add(IntToStr(SendId), Prev);

  RespRoot := nil;
  if JsonObjs.Count > 0 then
    RespRoot := TCnJSONObject(JsonObjs[0]);

  if RespRoot = nil then
  begin
    // һ��ԭʼ����
    Result := string(S);
  end
  else
  begin
    try
      // ������Ӧ
      HasPartly := False;
      if Partly then
      begin
        // ��ʽģʽ�£������ж�� Obj�������� type ������
        for I := 0 to JsonObjs.Count - 1 do
        begin
          // �� delta �е� text�����ж� type
          RespRoot := TCnJSONObject(JsonObjs[I]);
          if (RespRoot['delta'] <> nil) and (RespRoot['delta'] is TCnJSONObject) then
          begin
            Msg := TCnJSONObject(RespRoot['delta']);
            if (Msg['text'] <> nil) and (Msg['text'] is TCnJSONString) then
            begin
              // ÿһ���Ӧƴ����
              Result := Result + Msg['text'].AsString;
              HasPartly := True;
            end;
          end;
          if (RespRoot['content_block'] <> nil) and (RespRoot['content_block'] is TCnJSONObject) then
          begin
            Msg := TCnJSONObject(RespRoot['content_block']);
            if (Msg['text'] <> nil) and (Msg['text'] is TCnJSONString) then
            begin
              // ÿһ���Ӧƴ����
              Result := Result + Msg['text'].AsString;
              HasPartly := True;
            end;
          end;

          // Ҳ�ҽ������ type
          if (RespRoot['type'] <> nil) and (RespRoot['type'] is TCnJSONString) then
          begin
            if RespRoot['type'].AsString = 'message_stop' then
            begin
              IsStreamEnd := True;
              Exit;
            end;
          end;
        end;
      end
      else // ����ģʽ��
      begin
        // ������Ӧ
        if (RespRoot['content'] <> nil) and (RespRoot['content'] is TCnJSONArray) then
        begin
          Arr := TCnJSONArray(RespRoot['content']);
          if (Arr.Count > 0) and (Arr[0]['text'] <> nil) and (Arr[0]['text'] is TCnJSONString) then
            Result := Arr[0]['text'].AsString;
        end;

        if not HasPartly and (Result = '') then
        begin
          // ֻҪû��������Ӧ����˵�������ˣ��� Claude ���ĵ���û��˵����ֻ���������� AI ����д
          Success := False;

          // һ��ҵ����󣬱��� Key ��Ч��
          if (RespRoot['error'] <> nil) and (RespRoot['error'] is TCnJSONObject) then
          begin
            Msg := TCnJSONObject(RespRoot['error']);
            Result := Msg['message'].AsString;
          end;

          // һ��������󣬱��� URL ���˵�
          if (RespRoot['error'] <> nil) and (RespRoot['error'] is TCnJSONString) then
            Result := RespRoot['error'].AsString;
          if (RespRoot['message'] <> nil) and (RespRoot['message'] is TCnJSONString) then
          begin
            if Result = '' then
              Result := RespRoot['message'].AsString
            else
              Result := Result + ', ' + RespRoot['message'].AsString;
          end;
        end;
      end;

      // ���ף����н�������Ч��ֱ�������� JSON ��Ϊ������Ϣ
      if not HasPartly and (Result = '') then
        Result := string(S);
    finally
      JsonObjs.Free;
    end;
  end;

  // ����һ�»س�����
  if Pos(CRLF, Result) <= 0 then
    Result := StringReplace(Result, LF, CRLF, [rfReplaceAll]);
end;

procedure TCnClaudeAIEngine.PrepareRequestHeader(Headers: TStringList);
begin
  inherited;
  DeleteAuthorizationHeader(Headers); // ɾ��ԭ�е���֤ͷ������ͷ
  Headers.Add('x-api-key: ' + Option.ApiKey);
  Headers.Add('anthropic-version: ' + (Option as TCnClaudeAIEngineOption).AnthropicVersion);
end;

{ TCnGeminiAIEngine }

function TCnGeminiAIEngine.ConstructRequest(RequestType: TCnAIRequestType;
  const Code: string): TBytes;
var
  ReqRoot, Msg, Txt: TCnJSONObject;
  Cont, Part: TCnJSONArray;
  S: AnsiString;
begin
  ReqRoot := TCnJSONObject.Create;
  try
    Cont := ReqRoot.AddArray('contents');

    // Gemini ��֧�� system role��һ��� user ��
    Msg := TCnJSONObject.Create;
    Msg.AddPair('role', 'user');
    Part := Msg.AddArray('parts');
    Txt := TCnJSONObject.Create;

    if RequestType = artExplainCode then
      Txt.AddPair('text', Option.SystemMessage + #13#10 + Option.ExplainCodePrompt + #13#10 + Code)
    else if RequestType = artReviewCode then
      Txt.AddPair('text', Option.SystemMessage + #13#10 + Option.ReviewCodePrompt + #13#10 + Code)
    else if RequestType = artGenTestCase then
      Txt.AddPair('text', Option.SystemMessage + #13#10 + Option.GenTestCasePrompt + #13#10 + Code)
    else if RequestType = artRaw then
      Txt.AddPair('text', Option.SystemMessage + #13#10 + Code);

    Part.AddValue(Txt);
    Cont.AddValue(Msg);

    S := ReqRoot.ToJSON;
    Result := AnsiToBytes(S);
  finally
    ReqRoot.Free;
  end;
end;

class function TCnGeminiAIEngine.EngineName: string;
begin
  Result := 'Gemini';
end;

function TCnGeminiAIEngine.GetRequestURL(DataObj: TCnAINetRequestDataObject): string;
begin
  // ģ�����������֤�� Key ���� URL ��
  if Option.Stream then
    Result := DataObj.URL + Option.Model + ':streamGenerateContent?key=' + Option.ApiKey
  else
    Result := DataObj.URL + Option.Model + ':generateContent?key=' + Option.ApiKey;
end;

function TCnGeminiAIEngine.ParseResponse(SendId: Integer; StreamMode, Partly: Boolean;
  var Success: Boolean; var ErrorCode: Cardinal; var IsStreamEnd: Boolean;
  RequestType: TCnAIRequestType; const Response: TBytes): string;
var
  RespRoot, Cont, Msg: TCnJSONObject;
  Arr: TCnJSONArray;
  S, Prev: AnsiString;
  PrevS: string;
  P: PAnsiChar;
  HasPartly: Boolean;
  JsonObjs: TObjectList;
  I, Step: Integer;
begin
// ��ʽ��һ��ʼһ�� [��Ȼ��һ�� {},{},�����������{}���и� ]��
// ���Ǵ���ʱһ��ʼ���������� [ ���м�� , ������Ǹ� ] Ҫ���⴦��
// {
//  "candidates": [
//    {
//      "content": {
//        "parts": [
//          {
//            "text": "Application"
//          }
//        ],
//        "role": "model"
//      },
//      "finishReason": "STOP"
//    }
//  ],
//  "usageMetadata": {
//    "promptTokenCount": 46,
//    "totalTokenCount": 46
//  },
//  "modelVersion": "gemini-1.5-flash-latest"
// }

  Result := '';
  // ���� SendId �ұ��λỰ�����������
  Prev := '';
  if FPrevRespRemainMap.Find(IntToStr(SendId), PrevS) then
  begin
    Prev := AnsiString(PrevS);
    S := Prev + BytesToAnsi(Response) // ��ʣ������ƴ�����������ٴν��н���
  end
  else
    S := BytesToAnsi(Response);

  JsonObjs := TObjectList.Create(True);
  Step := CnJSONParse(PAnsiChar(S), JsonObjs);
  P := PAnsiChar(PAnsiChar(S) + Step);

  if (P <> #0) and (Step < Length(S)) then
  begin
    // ����û�����꣨����û���㣩���Ҳ��ǽ�������˵����ʣ������
    Prev := Copy(S, Step + 1, MaxInt);
  end
  else // ˵��ûʣ������
    Prev := '';

  // ����ʣ�඼������
  FPrevRespRemainMap.Add(IntToStr(SendId), Prev);

  RespRoot := nil;
  if JsonObjs.Count > 0 then
    RespRoot := TCnJSONObject(JsonObjs[0]);

  if RespRoot = nil then
  begin
    // һ��ԭʼ�������˺Ŵﵽ��󲢷��ȣ�ע����ģʽ��û�е����� datadone
    if Trim(S) <> ']' then // ������β�� ] Ҫ���ԣ�
      Result := string(S);
  end
  else
  begin
    try
      // ������Ӧ��Gemini ��ʽ
      HasPartly := False;
      if Partly then
      begin
        // ��ʽģʽ�£������ж�� Obj
        for I := 0 to JsonObjs.Count - 1 do
        begin
          RespRoot := TCnJSONObject(JsonObjs[I]);
          if (RespRoot['candidates'] <> nil) and (RespRoot['candidates'] is TCnJSONArray) then
          begin
            Arr := TCnJSONArray(RespRoot['candidates']);
            if (Arr.Count > 0) and (Arr[0]['content'] <> nil) and (Arr[0]['content'] is TCnJSONObject) then
            begin
              Cont := TCnJSONObject(Arr[0]['content']);
              if (Cont['parts'] <> nil) and (Cont['parts'] is TCnJSONArray) then
              begin
                Arr := TCnJSONArray(Cont['parts']);
                if (Arr.Count > 0) and (Arr[0]['text'] <> nil) and (Arr[0]['text'] is TCnJSONString) then
                begin
                  Msg := TCnJSONObject(Arr[0]);
                  Result := Result + Msg['text'].AsString;
                  HasPartly := True;
                end;
              end;
            end;

            // �� finishReason �ֶα�ʾ��β
            Arr := TCnJSONArray(RespRoot['candidates']);
            if (Arr.Count > 0) and (Arr[0]['finishReason'] <> nil) and (Arr[0]['finishReason'] is TCnJSONString) then
            begin
              IsStreamEnd := True;
              FPrevRespRemainMap.Delete(IntToStr(SendId)); // Ҳ�����棬����Ϊ�����ݣ�����ֱ�ӷ���
            end;
          end;
        end;
      end
      else // ����ģʽ
      begin
        if (RespRoot['candidates'] <> nil) and (RespRoot['candidates'] is TCnJSONArray) then
        begin
          Arr := TCnJSONArray(RespRoot['candidates']);
          if (Arr.Count > 0) and (Arr[0]['content'] <> nil) and (Arr[0]['content'] is TCnJSONObject) then
          begin
            Cont := TCnJSONObject(Arr[0]['content']);
            if (Cont['parts'] <> nil) and (Cont['parts'] is TCnJSONArray) then
            begin
              Arr := TCnJSONArray(Cont['parts']);
              if (Arr.Count > 0) and (Arr[0]['text'] <> nil) and (Arr[0]['text'] is TCnJSONString) then
              begin
                Msg := TCnJSONObject(Arr[0]);
                Result := Msg['text'].AsString;
              end;
            end;
          end;
        end;
      end;

      if not HasPartly and (Result = '') then
      begin
        // ֻҪû��������Ӧ����˵��������
        Success := False;

        // һ��ҵ����󣬱��� Key ��Ч��
        if (RespRoot['error'] <> nil) and (RespRoot['error'] is TCnJSONObject) then
        begin
          Msg := TCnJSONObject(RespRoot['error']);
          Result := Msg['message'].AsString;
        end;

        // һ��������󣬱��� URL ���˵�
        if (RespRoot['error'] <> nil) and (RespRoot['error'] is TCnJSONString) then
          Result := RespRoot['error'].AsString;
        if (RespRoot['message'] <> nil) and (RespRoot['message'] is TCnJSONString) then
        begin
          if Result = '' then
            Result := RespRoot['message'].AsString
          else
            Result := Result + ', ' + RespRoot['message'].AsString;
        end;
      end;

      // ���ף�����ģʽ�����н�������Ч��ֱ�������� JSON ��Ϊ������Ϣ
      if not HasPartly and (Result = '') then
        Result := string(S);
    finally
      RespRoot.Free;
    end;
  end;

  // ����һ�»س�����
  if Pos(CRLF, Result) <= 0 then
    Result := StringReplace(Result, LF, CRLF, [rfReplaceAll]);
end;

procedure TCnGeminiAIEngine.PrepareRequestHeader(Headers: TStringList);
begin
  inherited;
  // ɾԭ�е� Authorization����Ϊ�����֤�� URL ��
  DeleteAuthorizationHeader(Headers);
end;

{ TCnOllamaAIEngine }

function TCnOllamaAIEngine.ConstructRequest(RequestType: TCnAIRequestType;
  const Code: string): TBytes;
var
  ReqRoot, Msg: TCnJSONObject;
  Arr: TCnJSONArray;
  S: AnsiString;
begin
  ReqRoot := TCnJSONObject.Create;
  try
    ReqRoot.AddPair('model', Option.Model);
    ReqRoot.AddPair('stream', Option.Stream);
    Arr := ReqRoot.AddArray('messages');

    Msg := TCnJSONObject.Create;
    Msg.AddPair('role', 'system');
    Msg.AddPair('content', Option.SystemMessage);
    Arr.AddValue(Msg);

    Msg := TCnJSONObject.Create;
    Msg.AddPair('role', 'user');
    if RequestType = artExplainCode then
      Msg.AddPair('content', Option.ExplainCodePrompt + #13#10 + Code)
    else if RequestType = artReviewCode then
      Msg.AddPair('content', Option.ReviewCodePrompt + #13#10 + Code)
    else if RequestType = artGenTestCase then
      Msg.AddPair('content', Option.GenTestCasePrompt + #13#10 + Code)
    else if RequestType = artRaw then
      Msg.AddPair('content', Code);

    Arr.AddValue(Msg);

    S := ReqRoot.ToJSON;
    Result := AnsiToBytes(S);
  finally
    ReqRoot.Free;
  end;
end;

class function TCnOllamaAIEngine.EngineName: string;
begin
  Result := 'Ollama';
end;

class function TCnOllamaAIEngine.NeedApiKey: Boolean;
begin
  Result := False;
end;

function TCnOllamaAIEngine.ParseResponse(SendId: Integer; StreamMode, Partly: Boolean;
  var Success: Boolean; var ErrorCode: Cardinal; var IsStreamEnd: Boolean;
  RequestType: TCnAIRequestType; const Response: TBytes): string;
var
  RespRoot, Msg: TCnJSONObject;
  S, Prev: AnsiString;
  PrevS: string;
  P: PAnsiChar;
  HasPartly: Boolean;
  JsonObjs: TObjectList;
  I, Step: Integer;
begin
// ��ʽ��ʽ���£�
// {"model":"qwen2.5-coder","created_at":"2025-03-01T12:25:28.461904Z","message":
//   {"role":"assistant","content":"��"},"done":false}
// {"model":"qwen2.5-coder","created_at":"2025-03-01T12:25:29.178563Z","message":
//   {"role":"assistant","content":""},"done_reason":"stop","done":true,"
//   total_duration":104533692025,"load_duration":8214582022,"prompt_eval_count":35,
//   "prompt_eval_duration":21698000000,"eval_count":99,"eval_duration":73967000000}

  Result := '';
  // ���� SendId �ұ��λỰ�����������
  Prev := '';
  if FPrevRespRemainMap.Find(IntToStr(SendId), PrevS) then
  begin
    Prev := AnsiString(PrevS);
    S := Prev + BytesToAnsi(Response) // ��ʣ������ƴ�����������ٴν��н���
  end
  else
    S := BytesToAnsi(Response);

  JsonObjs := TObjectList.Create(True);
  Step := CnJSONParse(PAnsiChar(S), JsonObjs);
  P := PAnsiChar(PAnsiChar(S) + Step);

  if (P <> #0) and (Step < Length(S)) then
  begin
    // ����û�����꣨����û���㣩���Ҳ��ǽ�������˵����ʣ������
    Prev := Copy(S, Step + 1, MaxInt);
  end
  else // ˵��ûʣ������
    Prev := '';

  // ����ʣ�඼������
  FPrevRespRemainMap.Add(IntToStr(SendId), Prev);

  RespRoot := nil;
  if JsonObjs.Count > 0 then
    RespRoot := TCnJSONObject(JsonObjs[0]);

  if RespRoot = nil then
  begin
    // һ��ԭʼ�������˺Ŵﵽ��󲢷��ȣ�ע����ģʽ��û�е����� datadone
    Result := string(S);
  end
  else
  begin
    try
      // ������Ӧ
      HasPartly := False;
      if Partly then
      begin
        // ��ʽģʽ�£������ж�� Obj
        for I := 0 to JsonObjs.Count - 1 do
        begin
          RespRoot := TCnJSONObject(JsonObjs[I]);
          if (RespRoot['message'] <> nil) and (RespRoot['message'] is TCnJSONObject) then
          begin
            Msg := TCnJSONObject(RespRoot['message']);
            if  Msg['content'] <> nil then
              Result := Result + Msg['content'].AsString;

            HasPartly := True;
          end;

          // �� done �ֶ�Ϊ True ��ʾ��β
          if (RespRoot['done'] <> nil) and (RespRoot['done'] is TCnJSONTrue) then
          begin
            IsStreamEnd := True;
            FPrevRespRemainMap.Delete(IntToStr(SendId)); // Ҳ�����棬����Ϊ�����ݣ�����ֱ�ӷ���
          end;
        end;
      end
      else // ����ģʽ�����Ҫ��ʽ����ģʽ�е�����
      begin
        // Ollama �ļ�Ҫ��ʽ��message ����ʵ���� role: assistant ������
        if (RespRoot['message'] <> nil) and (RespRoot['message'] is TCnJSONObject) then
        begin
          Msg := TCnJSONObject(RespRoot['message']);
          if Msg['content'] <> nil then
            Result := Msg['content'].AsString;
        end;
      end;

      if not HasPartly and (Result = '') then
      begin
        // Ollama �ļ�Ҫ�����ʽ
        if (RespRoot['error'] <> nil) and (RespRoot['error'] is TCnJSONString) then
          Result := RespRoot['error'].AsString;
      end;

      // ���ף����н�������Ч��ֱ�������� JSON ��Ϊ������Ϣ
      if Result = '' then
        Result := string(S);
    finally
      JsonObjs.Free;
    end;
  end;

  // ����һ�»س�����
  if Pos(CRLF, Result) <= 0 then
    Result := StringReplace(Result, LF, CRLF, [rfReplaceAll]);
end;

initialization
  RegisterAIEngine(TCnDeepSeekAIEngine);
  RegisterAIEngine(TCnOpenAIAIEngine);
  RegisterAIEngine(TCnMistralAIAIEngine);
  RegisterAIEngine(TCnGeminiAIEngine);
  RegisterAIEngine(TCnClaudeAIEngine);
  RegisterAIEngine(TCnQWenAIEngine);
  RegisterAIEngine(TCnMoonshotAIEngine);
  RegisterAIEngine(TCnChatGLMAIEngine);
  RegisterAIEngine(TCnBaiChuanAIEngine);
  RegisterAIEngine(TCnOllamaAIEngine);

{$ENDIF CNWIZARDS_CNAICODERWIZARD}
end.
