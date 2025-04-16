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

unit CnSourceCropper;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�Դ��ע��ɾ������ģ��
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
* ��    ע��Դ��ע�ͽ���ģ��
* ����ƽ̨��Windows 2000 + Delphi 5
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2022.02.28 V1.2
*               ��ע��ǰ�������û�հף��Զ�����һ���ո�
*           2003.07.29 V1.1
*               ���ӱ����Զ����ʽע�͵Ĺ���
*           2003.06.15 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNCOMMENTCROPPERWIZARD}

uses
  Classes, SysUtils;

type
  TCnCropSourceTokenKind = (skUndefined, skCode, skBlockComment, skLineComment,
    skQuoteString, skDittoString, skDirective, skTodoList, skToReserve);

  TCnCropOption = (coAll, coExAscii);

type
  TCnSourceCropper = class
  private
    FCurTokenKind: TCnCropSourceTokenKind;
    FCurChar: AnsiChar;

    FCropTodoList: Boolean;
    FCropDirective: Boolean;
    FCropOption: TCnCropOption;
    FInStream: TStream;
    FOutStream: TStream;
    FReserve: Boolean;
    FReserveItems: TStringList;
    FRemoveSingleLineSlashes: Boolean;
    procedure SetInStream(const Value: TStream);
    procedure SetOutStream(const Value: TStream);
    procedure SetReserveItems(const Value: TStringList);
  protected
    procedure DoParse; virtual; abstract;
    procedure ProcessToBlockEnd; virtual; abstract;

    function IsTodoList: Boolean;
    function IsReserved: Boolean;
    function IsBlank(AChar: AnsiChar): Boolean;

    function GetCurChar: AnsiChar;
    function NextChar(Value: Integer = 1): AnsiChar;
    function PrevChar(Value: Integer = 1): AnsiChar;
    procedure WriteChar(Value: AnsiChar);
    procedure BackspaceChars(Values: Integer = 1); // �˸�ɾ������е� n ���ַ�
    procedure BackspaceOneCRLF; // �������е�ĩ���ַ��� CRLF ��ɾ��

    procedure ProcessToLineEnd(SpCount: Integer = 0; IsWholeLineSpace: Boolean = False); // ����Ĳ����Ǹ���ע��ǰ�������ո���
    procedure DoDefaultProcess;
    procedure DoBlockEndProcess;
  public
    procedure Parse;
    constructor Create; virtual;
    destructor Destroy; override;
  published
    property InStream: TStream read FInStream write SetInStream;
    {* ����Ҫ���� Ansi �� Utf8 ��ʽ�� AnsiString}
    property OutStream: TStream read FOutStream write SetOutStream;
    {* ������Ƕ�Ӧ�� Ansi �� Utf8 ��ʽ�� AnsiString}
    property CropOption: TCnCropOption read FCropOption write FCropOption;
    property CropDirective: Boolean read FCropDirective write FCropDirective;
    property CropTodoList: Boolean read FCropTodoList write FCropTodoList;
    property RemoveSingleLineSlashes: Boolean read FRemoveSingleLineSlashes write FRemoveSingleLineSlashes;
    {* �� // ע��ռ����������ʱ���Ƿ񽫸���һ��ɾ����ֻ��ȫɾ��ʱ��Ч}
    property Reserve: Boolean read FReserve write FReserve;
    property ReserveItems: TStringList read FReserveItems write SetReserveItems;
    {* �Ƿ����ض���ʽ��ע�� }
  end;

type
  TCnPasCropper = class(TCnSourceCropper)
  private

  protected
    procedure DoParse; override;
    procedure ProcessToBlockEnd; override;
    procedure ProcessToBracketBlockEnd;
  public

  published

  end;

type
  TCnCppCropper = class(TCnSourceCropper)
  private

  protected
    procedure DoParse; override;
    procedure ProcessToBlockEnd; override;
  public

  published

  end;

{$ENDIF CNWIZARDS_CNCOMMENTCROPPERWIZARD}

implementation

{$IFDEF CNWIZARDS_CNCOMMENTCROPPERWIZARD}

{ TCnSourceCropper }

const
  SCnToDo = 'TODO';
  SCnToDoDone = 'DONE';
  SCnCRLFSpacesChars: set of AnsiChar = [#0, #9, ' ', #13, #10];
  SCnSpacesChars: set of AnsiChar = [#9, ' '];
  SCnCRLFChars: set of AnsiChar = [#13, #10];

constructor TCnSourceCropper.Create;
begin
  inherited;
  FReserveItems := TStringList.Create;
end;

procedure TCnSourceCropper.BackspaceChars(Values: Integer);
begin
  if (OutStream <> nil) and (Values > 0) then
    if OutStream.Size > Values then
      OutStream.Size := OutStream.Size - Values;
end;

destructor TCnSourceCropper.Destroy;
begin
  FReserveItems.Free;
  inherited;
end;

procedure TCnSourceCropper.DoBlockEndProcess;
begin
  case FCurTokenKind of
  skBlockComment: // ����ע�ͣ�ֻ������ɾ����չ ASCII �������ַ�С�� 128 ��ʱ���д��
    if (FCropOption = coExAscii) and (FCurChar < #128) then
      WriteChar(FCurChar);
  skDirective: // ���ڱ���ָ�ֻ�в��������ָ���ʱ����д�������ʱ����ע������
    if not CropDirective or
      ((FCropOption = coExAscii) and (FCurChar < #128)) then
      WriteChar(FCurChar);
  skTodoList: // ���� todo��ֻ�в����� todo ��ʱ����д�������ʱ����ע������
     if not CropTodoList or
      ((FCropOption = coExAscii) and (FCurChar < #128)) then
      WriteChar(FCurChar);
  skToReserve:
    if FReserve then
      WriteChar(FCurChar);
  else
    DoDefaultProcess;
  end;
end;

procedure TCnSourceCropper.DoDefaultProcess;
begin
  if (FCropOption = coAll) or (FCurChar < #128) then
    WriteChar(FCurChar);
end;

// ��һ�ַ���ָ��ָ���һ����
function TCnSourceCropper.GetCurChar: AnsiChar;
begin
  Result := #0;
  if Assigned(FInStream) then
  begin
    try
      FInStream.Read(Result, SizeOf(AnsiChar));
    except
      Exit;
    end;
  end;
end;

function TCnSourceCropper.IsBlank(AChar: AnsiChar): Boolean;
begin
  Result := AChar in [' ', #13, #10, #7, #9];
end;

function TCnSourceCropper.IsReserved: Boolean;
var
  I: Integer;
  OldChar: AnsiChar;
  OldPos: Integer;
  MaxLen: Integer;
  PBuf: PChar;
  SToCompare: String;
begin
  // �ж��Ƿ����ڱ����б��еĶ�����Ҳ�����ж��Ƿ�Ӧ�ñ���
  Result := False;
  if FInStream = nil then Exit;

  PBuf := nil;
  OldChar := FCurChar;
  OldPos := FInStream.Position;

  MaxLen := 0;
  for I := FReserveItems.Count - 1 downto 0 do
  begin
    if MaxLen < Length(FReserveItems.Strings[I]) then
      MaxLen := Length(FReserveItems.Strings[I]);
    if FReserveItems.Strings[I] = '' then
      FReserveItems.Delete(I);
  end;

  if (FCurChar = '/') or (FCurChar = '(') then
  begin
    FCurChar := GetCurChar;
    if FCurChar <> '*' then
      Exit;
  end;
  // ��ʱ FCurChar ָ��ע�Ϳ�ʼ���ŵ����һ�ֽڣ�{ �� /* �� * �� (* �� *

  try
    PBuf := StrAlloc(MaxLen + 1);
    FillChar(PBuf^, Length(PBuf), 0);
    FInStream.Read(PBuf^, MaxLen);

    for I := 0 to FReserveItems.Count - 1 do
    begin
      SToCompare := Copy(StrPas(PBuf), 1, Length(FReserveItems.Strings[I]));
      if SToCompare = FReserveItems.Strings[I] then
      begin
        Result := True;
        Exit;
      end;
    end;
  finally
    FCurChar := OldChar;
    FInStream.Position := OldPos;
    if PBuf <> nil then
      StrDispose(PBuf);
  end;
end;

function TCnSourceCropper.IsTodoList: Boolean;
var
  OldPos: Integer;
  OldChar: AnsiChar;
  PTodo: PChar;
  STodo: String;
begin
  // (* �� { �� // �����޿հ��� Todo �� Done�������޿ո��ð�ţ����ǺϷ��� TodoList.
  // ����ʱ��FCurChar ������ '{'���� '(*' �� '('���� '/*' �е� '/'���� '//' �еĵ�һ�� '/'��

  Result := False;
  if FInStream = nil then Exit;

  PTodo := nil;
  OldChar := FCurChar;
  OldPos := FInStream.Position;
  try
    if (FCurChar = '/') or (FCurChar = '(') then
    begin
      FCurChar := GetCurChar;
      if (FCurChar <> '*') and (FCurChar <> '/') then
        Exit;
    end;
    // ��ʱ FCurChar ָ��ע�Ϳ�ʼ���ŵ����һ�ֽڣ�{ �� /* �� * �� (* �� * �� // �ĵڶ��� /

    while IsBlank(NextChar) do
      FCurChar := GetCurChar;
    // ��ʱ FCurChar ָ��ע���в�Ϊ�յĵ�һ���ַ���ǰһ�ַ���������{��*���

    PTodo := StrAlloc(Length(SCnToDo) + 1);
    FillChar(PTodo^, Length(PTodo), 0);
    FInStream.Read(PTodo^, Length(SCnToDo));
    STodo := Copy(UpperCase(StrPas(PTodo)), 1, 4);

    if (STodo = SCnTodo) or (STodo = SCnTodoDone) then
    begin
      // ��ʱָ��ָ�� todo ��һ���ַ���
      while IsBlank(NextChar) do
        FCurChar := GetCurChar;
        
      if NextChar = ':' then
      begin
        Result := True;
        Exit;
      end
    end;

  finally
    FCurChar := OldChar;
    FInStream.Position := OldPos;
    if PTodo <> nil then
      StrDispose(PTodo);
  end;
end;

// ��һ�ַ���ָ��λ�ò��䣬��Ȼ�ڵ�ǰ�ַ��ĺ�һλ�á�
function TCnSourceCropper.NextChar(Value: Integer): AnsiChar;
begin
  Result := #0;
  if Assigned(FInStream) then
  begin
    try
      FInStream.Seek(Value - 1, soFromCurrent);
      FInStream.Read(Result, SizeOf(AnsiChar));
      FInStream.Seek(-Value, soFromCurrent);
    except
      Exit;
    end;
  end;
end;

procedure TCnSourceCropper.Parse;
begin
  if (FInStream <> nil) and (FOutStream <> nil) then
  begin
    if (FInStream.Size > 0) then
    begin
      FInStream.Position := 0;
      FCurTokenKind := skUndefined;
      DoParse;
    end;
  end;
end;

function TCnSourceCropper.PrevChar(Value: Integer): AnsiChar;
begin
  Result := #0;
  if Assigned(FInStream) then
  begin
    try
      if FInStream.Position - Value - 1 >= 0 then // ǰ�����㹻��λ��������
      begin
        FInStream.Seek(- Value - 1, soFromCurrent);
        FInStream.Read(Result, SizeOf(AnsiChar));
        FInStream.Seek(Value, soFromCurrent);
      end;
    except
      Exit;
    end;
  end;
end;

procedure TCnSourceCropper.ProcessToLineEnd(SpCount: Integer; IsWholeLineSpace: Boolean);
begin
  if (FCropOption = coAll) and (FCurTokenKind <> skTodoList) then
  begin
    BackspaceChars(SpCount);
    if IsWholeLineSpace and FRemoveSingleLineSlashes then
      BackspaceOneCRLF;

    while not (FCurChar in [#0, #13, #10]) do
      FCurChar := GetCurChar;
  end
  else
  begin
    while not (FCurChar in [#0, #13, #10]) do
    begin
      if ((FCropOption = coExAscii) and (FCurChar < #128))
        or (FCurTokenKind = skTodoList) then
          WriteChar(FCurChar);
      FCurChar := GetCurChar;
    end;
  end;

  // ��ǰ�� #13 �� #10
  if FCurChar = #13 then
  begin
    repeat
      WriteChar(FCurChar);   // �س�����Ҫд�ġ�
      FCurChar := GetCurChar;
    until FCurChar in [#0, #10];
  end;

  if FCurChar = #10 then
    WriteChar(FCurChar);

  // ���غ�FCurChar ָ�� #10 �� #0��Ҳ�������һ����������ַ���
  FCurTokenKind := skUndefined;
end;

procedure TCnSourceCropper.SetInStream(const Value: TStream);
begin
  FInStream := Value;
end;

procedure TCnSourceCropper.SetOutStream(const Value: TStream);
begin
  FOutStream := Value;
end;

procedure TCnSourceCropper.SetReserveItems(const Value: TStringList);
begin
  if Value <> nil then
    FReserveItems.Assign(Value);
end;

procedure TCnSourceCropper.WriteChar(Value: AnsiChar);
begin
  if Assigned(FOutStream) then
  begin
    try
      OutStream.Write(Value, SizeOf(Value));
    except
      Exit;
    end;
  end;
end;

procedure TCnSourceCropper.BackspaceOneCRLF;
var
  C: AnsiChar;
begin
  if (OutStream.Size <= 0) or (OutStream.Position <= 0) then // ǰ��û����
    Exit;

  OutStream.Seek(-1, soFromCurrent);
  OutStream.Read(C, SizeOf(AnsiChar));
  OutStream.Seek(1, soFromCurrent);

  if C = #10 then
  begin
    OutStream.Size := OutStream.Size - 1;
    if (OutStream.Size <= 0) or (OutStream.Position <= 0) then // ǰ��û����
      Exit;

    OutStream.Seek(-1, soFromCurrent);
    OutStream.Read(C, SizeOf(AnsiChar));
    OutStream.Seek(1, soFromCurrent);
    if C = #13 then
      OutStream.Size := OutStream.Size - 1;
  end;
end;

{ TCnCppCropper }

procedure TCnCppCropper.DoParse;
var
  IsSpace, WholeLineSpace: Boolean;
  SpCount: Integer;
begin
  FCurChar := GetCurChar;
  SpCount := 0;
  WholeLineSpace := True;

  while FCurChar <> #0 do
  begin
    case FCurChar of
    '/':
      begin
        if (FCurTokenKind in [skCode, skUndefined]) and (NextChar = '/') then
        begin
          if IsTodoList then
            FCurTokenKind := skTodoList
          else
            FCurTokenKind := skLineComment;
          // ���Ŵ�����β��
          ProcessToLineEnd(SpCount, WholeLineSpace);
        end
        else
        if (FCurTokenKind in [skCode, skUndefined]) and (NextChar = '*') then
        begin
          // ����Ƿ��� TodoList
          if IsTodoList then
            FCurTokenKind := skTodoList
          else if FReserve and IsReserved then  // (NextChar(2) = '#')
            FCurTokenKind := skToReserve
          else
            FCurTokenKind := skBlockComment;
          // ���� '*/'
          ProcessToBlockEnd;
        end
        else
          DoDefaultProcess;
      end;
    '''':
      begin
        if FCurTokenKind in [skCode, skUndefined] then
          FCurTokenKind := skQuoteString
        else if FCurTokenKind = skQuoteString then
           FCurTokenKind := skCode;

        DoDefaultProcess;       
      end;
    '"':
      begin
        if FCurTokenKind in [skCode, skUndefined] then
          FCurTokenKind := skDittoString
        else if FCurTokenKind = skDittoString then
           FCurTokenKind := skCode;

        DoDefaultProcess;
      end;
    else
      DoDefaultProcess;
    end;

    IsSpace := FCurChar in SCnSpacesChars;
    if IsSpace then
      Inc(SpCount)
    else
    begin
      SpCount := 0;
      if not (FCurChar in SCnCRLFSpacesChars) then
        WholeLineSpace := False;
    end;

    if FCurChar in SCnCRLFChars then
      WholeLineSpace := True;

    FCurChar := GetCurChar;
  end;
  WriteChar(#0);
end;

procedure TCnCppCropper.ProcessToBlockEnd;
var
  NeedSep: Boolean;
begin
  NeedSep := not (PrevChar in SCnCRLFSpacesChars);    // ��¼��ǰ���޿հ�

  while ((FCurChar <> '*') or (NextChar <> '/')) and (FCurChar <> #0) do
  begin
    DoBlockEndProcess;
    FCurChar := GetCurChar;
  end;

  // ��ʱ FCurChar �Ѿ�ָ���� '*'�����Һ������ '/'��
  if FCurChar = '*' then
  begin
    DoBlockEndProcess;   // д *
    FCurChar := GetCurChar;
    DoBlockEndProcess;   // д /
  end;

  FCurTokenKind := skUndefined;
  // ���ַ��Ѿ�������д����

  if NeedSep and not (FCurChar in SCnCRLFSpacesChars) then // �����ǰ���û�հף���д���ո�������
    WriteChar(' ');
end;

{ TCnPasCropper }

procedure TCnPasCropper.DoParse;
var
  IsSpace, WholeLineSpace: Boolean;
  SpCount: Integer;
begin
  FCurChar := GetCurChar;
  SpCount := 0;
  WholeLineSpace := True;

  while FCurChar <> #0 do
  begin
    case FCurChar of
    '/':
      begin
        if (FCurTokenKind in [skCode, skUndefined]) and (NextChar = '/') then
        begin
          if IsTodoList then
            FCurTokenKind := skTodoList
          else
            FCurTokenKind := skLineComment;
          // ���Ŵ�����β��
          ProcessToLineEnd(SpCount, WholeLineSpace);
        end
        else
          DoDefaultProcess;
      end;
    '{':
      begin
        if FCurTokenKind in [skCode, skUndefined] then
        begin
          if NextChar <> '$' then
          begin
            // ����Ƿ��� TodoList
            if IsTodoList then
              FCurTokenKind := skTodoList
            else if FReserve and IsReserved then      // (NextChar = '*')
              FCurTokenKind := skToReserve
            else
              FCurTokenKind := skBlockComment
          end
          else
            FCurTokenKind := skDirective;
          // ���� '}' �š�
          ProcessToBlockEnd;
        end
        else
          DoDefaultProcess;
      end;
    '(':
      begin
        if (FCurTokenKind in [skCode, skUndefined]) and (NextChar = '*') then
        begin
          // ����Ƿ��� TodoList
          if IsTodoList then
            FCurTokenKind := skTodoList
          else if NextChar(2) = '$' then
            FCurTokenKind := skDirective
          else
            FCurTokenKind := skBlockComment;
          // ���� '*)'
          ProcessToBracketBlockEnd;
        end
        else
          DoDefaultProcess;
      end;
    '''':
      begin
        if FCurTokenKind in [skCode, skUndefined] then
          FCurTokenKind := skQuoteString
        else if FCurTokenKind = skQuoteString then
           FCurTokenKind := skCode;

        DoDefaultProcess;
      end;
    else
      DoDefaultProcess;
    end;

    IsSpace := FCurChar in SCnSpacesChars;
    if IsSpace then
      Inc(SpCount)
    else
    begin
      SpCount := 0;
      if not (FCurChar in SCnCRLFSpacesChars) then
        WholeLineSpace := False;
    end;

    if FCurChar in SCnCRLFChars then
      WholeLineSpace := True;

    FCurChar := GetCurChar;
  end;
  WriteChar(#0);
end;

procedure TCnPasCropper.ProcessToBlockEnd;
var
  NeedSep: Boolean;
begin
  NeedSep := not (PrevChar in SCnCRLFSpacesChars);  // ��¼��ǰ���޿հ�

  while not (FCurChar in [#0, '}']) do
  begin
    DoBlockEndProcess;
    FCurChar := GetCurChar;
  end;

  // ��ʱ FCurChar �Ѿ�ָ�������� '}'��Ҳ�������һ����������ַ�
  DoBlockEndProcess;
  FCurTokenKind := skUndefined;
  // ���ַ��Ѿ�������д����

  if NeedSep and not (FCurChar in SCnCRLFSpacesChars) then // �����ǰ���û�հף���д���ո�������
    WriteChar(' ');
end;

procedure TCnPasCropper.ProcessToBracketBlockEnd;
var
  NeedSep: Boolean;
begin
  NeedSep := not (PrevChar in SCnCRLFSpacesChars);  // ��¼��ǰ���޿հ�

  while ((FCurChar <> '*') or (NextChar <> ')')) and (FCurChar <> #0) do
  begin
    DoBlockEndProcess;
    FCurChar := GetCurChar;
  end;

  // ��ʱ FCurChar �Ѿ�ָ���� '*'�����Һ������ ')'��
  if FCurChar = '*' then
  begin
    DoBlockEndProcess;   // д *
    FCurChar := GetCurChar;
    DoBlockEndProcess;   // д )
  end;

  FCurTokenKind := skUndefined;
  // ���ַ��Ѿ�������д����

  if NeedSep and not (FCurChar in SCnCRLFSpacesChars) then // �����ǰ���û�հף���д���ո�������
    WriteChar(' ');
end;

{$ENDIF CNWIZARDS_CNCOMMENTCROPPERWIZARD}
end.
