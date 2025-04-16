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

unit CnWizDfmParser;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ����� DFM �ļ���Ϣ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWinXP SP2 + Delphi 7.1
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2012.09.19 by shenloqi
*               ��ֲ�� Delphi XE3
*           2005.03.23 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, SysUtils, Classes, CnCommon, CnTree,
{$IFDEF COMPILER6_UP}
  Variants, RTLConsts,
{$ELSE}
  Consts,
{$ENDIF}
  TypInfo;

type
  TDfmFormat = (dfUnknown, dfText, dfBinary);
  TDfmKind = (dkObject, dkInherited, dkInline);

  TDfmInfo = class(TPersistent)
  private
    FFormat: TDfmFormat;
    FKind: TDfmKind;
    FName: string;
    FFormClass: string;
    FCaption: string;
    FLeft: Integer;
    FTop: Integer;
    FWidth: Integer;
    FHeight: Integer;
  published
    property Top: Integer read FTop write FTop;
    property Width: Integer read FWidth write FWidth;
    property Name: string read FName write FName;
    property Left: Integer read FLeft write FLeft;
    property Kind: TDfmKind read FKind write FKind;
    property Height: Integer read FHeight write FHeight;
    property Format: TDfmFormat read FFormat write FFormat;
    property FormClass: string read FFormClass write FFormClass;
    property Caption: string read FCaption write FCaption;
  end;

  TCnDfmTree = class;

  TCnDfmLeaf = class(TCnLeaf)
  {* ���� DFM �е�һ�����}
  private
    FElementClass: string;
    FElementKind: TDfmKind;
    FProperties: TStrings;
    function GetItems(Index: Integer): TCnDfmLeaf;
    procedure SetItems(Index: Integer; const Value: TCnDfmLeaf);
    function GetTree: TCnDfmTree;
    function GetPropertyValue(const PropertyName: string): string;
  protected
    procedure AssignTo(Dest: TPersistent); override;
  public
    constructor Create(ATree: TCnTree); override;
    {* ���췽�� }
    destructor Destroy; override;
    {* �������� }

    procedure AppendToStrings(List: TStrings; Tab: Integer = 0);
    {* ���������֡������Լ���������д��һ���ַ����б��������ã���дĩβ�� end}

    property Items[Index: Integer]: TCnDfmLeaf read GetItems write SetItems; default;
    property Tree: TCnDfmTree read GetTree;

    // ע�⣺���������У��ø���� Text ���Ե��� Name
    property ElementClass: string read FElementClass write FElementClass;
    {* ClassName ����}
    property ElementKind: TDfmKind read FElementKind write FElementKind;
    {* Ԫ������}
    property Properties: TStrings read FProperties;
    {* �洢�ı����ԣ���ʽΪ PropName = PropValue�����ڸ������ԣ�PropValue ����ܰ����س�
      ע�� Objects ��������ܴ� TMemoryStream �Ķ���������}
    property PropertyValue[const PropertyName: string]: string read GetPropertyValue;
    {* ����ĳ�������õ�����ֵ��������ַ�������� DFM �еĵ�������ת�룬ע�ⲻ֧�ֶ�������}
  end;

  TCnDfmTree = class(TCnTree)
  {* ���� DFM �������� Root �ĵ�һ���ӽڵ��Ǹ�������������ʵ�����}
  private
    FDfmFormat: TDfmFormat;
    FDfmKind: TDfmKind;
    function GetRoot: TCnDfmLeaf;
    function GetItems(AbsoluteIndex: Integer): TCnDfmLeaf;
  protected
    procedure SaveLeafToStrings(Leaf: TCnDfmLeaf; List: TStrings; Tab: Integer = 0);
  public
    constructor Create;
    destructor Destroy; override;

    function SaveToStrings(List: TStrings): Boolean;
    function GetSameClassIndex(Leaf: TCnDfmLeaf): Integer;
    {* �ں͸� Leaf �� ElementClass ��ͬ�� Leaf ���Ҹ� Leaf �� Index��0 ��ʼ}

    property Root: TCnDfmLeaf read GetRoot;
    property Items[AbsoluteIndex: Integer]: TCnDfmLeaf read GetItems;

    property DfmKind: TDfmKind read FDfmKind write FDfmKind;
    property DfmFormat: TDfmFormat read FDfmFormat write FDfmFormat;
  end;

const
  SDfmFormats: array[TDfmFormat] of string = ('Unknown', 'Text', 'Binary');
  SDfmKinds: array[TDfmKind] of string = ('Object', 'Inherited', 'Inline');

function ParseDfmStream(Stream: TStream; Info: TDfmInfo): Boolean;
{* �򵥽��� DFM ����������� Container ����Ϣ}

function ParseDfmFile(const FileName: string; Info: TDfmInfo): Boolean;
{* �򵥽��� DFM �ļ���������� Container ����Ϣ}

function LoadMultiTextStreamToTree(Stream: TStream; Tree: TCnDfmTree): Boolean;
{* ���ַ��������������������ε�����Ӧ�Լ������������޸��Ķ�����
  ע�� Stream �ڵ�������Ҫ�� AnsiString����Ϊ TParser û������ UTF16 ��}

function LoadDfmStreamToTree(Stream: TStream; Tree: TCnDfmTree): Boolean;
{* �� DFM ����������}

function LoadDfmFileToTree(const FileName: string; Tree: TCnDfmTree): Boolean;
{* �� DFM �ļ���������}

function SaveTreeToDfmFile(const FileName: string; Tree: TCnDfmTree): Boolean;
{* ���������ݴ�� DFM �ı��ļ�}

function SaveTreeToStrings(const List: TStrings; Tree: TCnDfmTree): Boolean;
{* ���������ݴ���ַ����б�}

function ConvertWideStringToDfmString(const W: WideString): WideString;
{* �����ַ���ת��Ϊ Delphi 7 �����ϰ汾�е� DFM �ַ���}

function ConvertStreamToHexDfmString(Stream: TStream; Tab: Integer = 2): string;
{* ������������ת��Ϊ DFM �ַ���������ǰ�������}

function DecodeDfmStr(const QuotedStr: string): string;
{* �� Caption �����ִ����ŵ�ת�����ַ�ת��Ϊ�����ַ���}

implementation

const
  csPropCount = 5;
  csTabWidth = 2;
  CRLF = #13#10;
  FILER_SIGNATURE: array[1..4] of AnsiChar = ('T', 'P', 'F', '0');

{$IFNDEF COMPILER6_UP}
function CombineString(Parser: TParser): string;
begin
  Result := Parser.TokenString;
  while Parser.NextToken = '+' do
  begin
    Parser.NextToken;
    Parser.CheckToken(toString);
    Result := Result + Parser.TokenString;
  end;
end;
{$ENDIF}

function ConvertStreamToHexDfmString(Stream: TStream; Tab: Integer): string;
const
  BYTES_PER_LINE = 32;
var
  I, Count: Integer;
  MultiLine: Boolean;
  Buffer: array[0..BYTES_PER_LINE - 1] of AnsiChar;
  Text: array[0..BYTES_PER_LINE * 2 - 1] of Char;
begin
  Result := '';
  Count := Stream.Size;
  MultiLine := Count >= BYTES_PER_LINE;
  if Tab < 0 then
    Tab := 0;

  while Count > 0 do
  begin
    if MultiLine then
      Result := Result + #13#10 + StringOfChar(' ', Tab);

    if Count >= BYTES_PER_LINE then
      I := BYTES_PER_LINE
    else
      I := Count;

    Stream.Read(Buffer, I);
    BinToHex(Buffer, Text, I);
    Result := Result + Copy(Text, 1, I * 2);
    Dec(Count, I);
  end;
end;

function ConvertWideStringToDfmString(const W: WideString): WideString;
const
  LINE_LENGTH = 1024;
var
  L, I, J, K: Integer;
  LineBreak: Boolean;
begin
  L := Length(W);

  if L = 0 then
    Result := ''''''
  else
  begin
    I := 1;
    if L > LINE_LENGTH then
      Result := Result + #13#10;
    K := I;
    repeat
      LineBreak := False;
      if (W[I] >= ' ') and (W[I] <> '''') and (Ord(W[I]) <= 127) then
      begin
        J := I;
        repeat
          Inc(I)
        until (I > L) or (W[I] < ' ') or (W[I] = '''') or
          ((I - K) >= LINE_LENGTH) or (Ord(W[I]) > 127);
        if ((I - K) >= LINE_LENGTH) then
          LineBreak := True;
        Result := Result + '''';
        while J < I do
        begin
          Result := Result + Char(W[J]);
          Inc(J);
        end;
        Result := Result + '''';
      end else
      begin
        Result := Result + '#' + IntToStr(Ord(W[I]));
        Inc(I);
        if ((I - K) >= LINE_LENGTH) then
          LineBreak := True;
      end;
      if LineBreak and (I <= L) then
      begin
        Result := Result + ' +' + #13#10;
        K := I;
      end;
    until I > L;
  end;
end;

function CombineWideString(Parser: TParser): WideString;
begin
  Result := Parser.TokenWideString;
  while Parser.NextToken = '+' do
  begin
    Parser.NextToken;
    if not CharInSet(Parser.Token, [toString, toWString]) then
      Parser.CheckToken(toString);
    Result := Result + Parser.TokenWideString;
  end;
end;

function ParseTextOrderModifier(Parser: TParser): Integer;
begin
  Result := -1;
  if Parser.Token = '[' then
  begin
    Parser.NextToken;
    Parser.CheckToken(toInteger);
    Result := Parser.TokenInt;
    Parser.NextToken;
    Parser.CheckToken(']');
    Parser.NextToken;
  end;
end;

function ParseTextPropertyValue(Parser: TParser; out BinStream: TObject): string; forward;

procedure ParseTextHeaderToLeaf(Parser: TParser; IsInherited, IsInline: Boolean;
  Leaf: TCnDfmLeaf); forward;

procedure ParseTextPropertyToLeaf(Parser: TParser; Leaf: TCnDfmLeaf);
var
  PropName: string;
  PropValue: string;
  Obj: TObject;
begin
  Parser.CheckToken(toSymbol);
  PropName := Parser.TokenString;
  Parser.NextToken;
  while Parser.Token = '.' do
  begin
    Parser.NextToken;
    Parser.CheckToken(toSymbol);
    PropName := PropName + '.' + Parser.TokenString;
    Parser.NextToken;
  end;

  Parser.CheckToken('=');
  Parser.NextToken;
  Obj := nil;
  PropValue := ParseTextPropertyValue(Parser, Obj);

  if Obj <> nil then
    Leaf.Properties.AddObject(PropName + ' = ' + PropValue, Obj)
  else
    Leaf.Properties.Add(PropName + ' = ' + PropValue);
end;

function ParseTextPropertyValue(Parser: TParser; out BinStream: TObject): string;
var
  Stream: TStream;
  QS: string;

  function GetQuotedStr: string;
  begin
{$IFDEF COMPILER6_UP}
    if CharInSet(Parser.Token, [toString, toWString]) then
    begin
      Result := CombineWideString(Parser);
      Result := ConvertWideStringToDfmString(Result);
    end;
{$ELSE}
    // ����ƴ��һ�еĸ����ã��������Ȳ���
    if Parser.Token = toString then
      Result := QuotedStr(CombineString(Parser))
    else if Parser.Token = toWString then
      Result := QuotedStr(CombineWideString(Parser));
{$ENDIF}
  end;

begin
  Result := '';
{$IFDEF COMPILER6_UP}
  if CharInSet(Parser.Token, [toString, toWString]) then
  begin
    Result := CombineWideString(Parser);
    Result := ConvertWideStringToDfmString(Result);
  end
{$ELSE}
  // ����ƴ��һ�еĸ����ã��������Ȳ���
  if Parser.Token = toString then
    Result := QuotedStr(CombineString(Parser))
  else if Parser.Token = toWString then
    Result := QuotedStr(CombineWideString(Parser))
{$ENDIF}
  else
  begin
    case Parser.Token of
      toSymbol:
        Result := Parser.TokenComponentIdent;
      toInteger:
        Result := IntToStr(Parser.TokenInt);
      toFloat:
        Result := FloatToStr(Parser.TokenFloat);
      '[':  // ����
        begin
          Result := Parser.TokenString;
          Parser.NextToken;
          while Parser.Token <> ']' do
          begin
            if Parser.Token = ',' then
              Result := Result + Parser.TokenString + ' '
            else
              Result := Result + Parser.TokenString;
            Parser.NextToken;
          end;
          Result := Result + ']';
        end;
      '(':  // �ַ����б�� DesignSize �������б����������ʱ���ƣ����ﲻ������
        begin
          Result := Parser.TokenString;
          Parser.NextToken;
          while Parser.Token <> ')' do
          begin
            QS := GetQuotedStr;
            if QS <> '' then
              Result := Result + #13#10 + '  ' + QS
            else
              Parser.NextToken; // GetQuotedStr �ڲ��Ѿ� NextToken �ˣ����������к���
          end;
          Result := Result + ')';
        end;
      '{':  // ����������
        begin
          Result := Parser.TokenString;
          // Parser.NextToken; // ���� NextToken������ HexToBinary ������һ��

          Stream := TMemoryStream.Create;
          Parser.HexToBinary(Stream);
          Stream.Position := 0;
          Result := ConvertStreamToHexDfmString(Stream) + '}';

          BinStream := Stream; // �����ж��������ݵ������󴫳�
        end;
      '<':  // TODO: Collection �� Items ��Ҫ�ָ��
        begin
          Result := Parser.TokenString;
          Parser.NextToken;
          while Parser.Token <> '>' do
          begin
            Result := Result + Parser.TokenString;
            Parser.NextToken;
          end;
          Result := Result + '>';
        end;
    else
      Parser.Error(SInvalidProperty);
    end;
    Parser.NextToken;
  end;
end;

// �ݹ���� Object���������ʱ Parser ͣ���� object��Leaf �Ǹ��½���
procedure ParseTextObjectToLeaf(Parser: TParser; Tree: TCnDfmTree; Leaf: TCnDfmLeaf);
var
  InheritedObject: Boolean;
  InlineObject: Boolean;
  Child: TCnDfmLeaf;
begin
  InheritedObject := False;
  InlineObject := False;
  if Parser.TokenSymbolIs('INHERITED') then
  begin
    InheritedObject := True;
    Leaf.ElementKind := dkInherited;
  end
  else if Parser.TokenSymbolIs('INLINE') then
  begin
    InlineObject := True;
    Leaf.ElementKind := dkInline;
  end
  else
  begin
    Parser.CheckTokenSymbol('OBJECT');
    Leaf.ElementKind := dkObject;
  end;

  Parser.NextToken;
  ParseTextHeaderToLeaf(Parser, InheritedObject, InlineObject, Leaf);

  while not Parser.TokenSymbolIs('END') and
    not Parser.TokenSymbolIs('OBJECT') and
    not Parser.TokenSymbolIs('INHERITED') and
    not Parser.TokenSymbolIs('INLINE') do
    ParseTextPropertyToLeaf(Parser, Leaf);

  while Parser.TokenSymbolIs('OBJECT') or
    Parser.TokenSymbolIs('INHERITED') or
    Parser.TokenSymbolIs('INLINE') do
  begin
    Child := Tree.AddChild(Leaf) as TCnDfmLeaf;
    ParseTextObjectToLeaf(Parser, Tree, Child);
  end;
  Parser.NextToken; // �� end
end;

procedure ParseTextHeaderToLeaf(Parser: TParser; IsInherited, IsInline: Boolean; Leaf: TCnDfmLeaf);
begin
  Parser.CheckToken(toSymbol);
  Leaf.ElementClass := Parser.TokenString;
  Leaf.Text := '';
  if Parser.NextToken = ':' then
  begin
    Parser.NextToken;
    Parser.CheckToken(toSymbol);
    Leaf.Text := Leaf.ElementClass;
    Leaf.ElementClass := Parser.TokenString;
    Parser.NextToken;
  end;
  ParseTextOrderModifier(Parser);
end;

procedure ParseBinaryHeader(Reader: TReader; Leaf: TCnDfmLeaf);
var
  Flags: TFilerFlags;
  Position: Integer;
begin
  Reader.ReadPrefix(Flags, Position);
  Leaf.ElementClass := Reader.ReadStr;
  Leaf.Text := Reader.ReadStr;
  if Leaf.Text = '' then
    Leaf.Text := Leaf.ElementClass;
end;

procedure ParseBinaryObjectToLeaf(Reader: TReader; Leaf: TCnDfmLeaf);
begin
  ParseBinaryHeader(Reader, Leaf);
  // TODO: Parse Binary Properties and Children to Leaves.
//  while not Reader.EndOfList do
//    ParseBinaryPropertyToLeaf(True);
end;

// �򵥽��� Text ��ʽ�� Dfm �õ� Info
function ParseTextDfmStream(Stream: TStream; Info: TDfmInfo): Boolean;
var
  SaveSeparator: Char;
  Parser: TParser;
  PropCount: Integer;

  procedure ParseHeader(IsInherited, IsInline: Boolean);
  begin
    Parser.CheckToken(toSymbol);
    Info.FormClass := Parser.TokenString;
    Info.Name := '';
    if Parser.NextToken = ':' then
    begin
      Parser.NextToken;
      Parser.CheckToken(toSymbol);
      Info.Name := Info.FormClass;
      Info.FormClass := Parser.TokenString;
      Parser.NextToken;
    end;
    ParseTextOrderModifier(Parser);
  end;

  procedure ParseProperty(IsForm: Boolean); forward;

  function ParseValue: Variant;
  begin
    Result := Null;
  {$IFDEF COMPILER6_UP}
    if CharInSet(Parser.Token, [toString, toWString]) then
      Result := CombineWideString(Parser)
  {$ELSE}
    if Parser.Token = toString then
      Result := CombineString(Parser)
    else if Parser.Token = toWString then
      Result := CombineWideString(Parser)
  {$ENDIF}
    else
    begin
      case Parser.Token of
        toSymbol:
          Result := Parser.TokenComponentIdent;
        toInteger:
        {$IFDEF COMPILER6_UP}
          Result := Parser.TokenInt;
        {$ELSE}
          Result := Integer(Parser.TokenInt);
        {$ENDIF}
        toFloat:
          Result := Parser.TokenFloat;
        '[':
          begin
            Parser.NextToken;
            if Parser.Token <> ']' then
              while True do
              begin
                if Parser.Token <> toInteger then
                  Parser.CheckToken(toSymbol);
                if Parser.NextToken = ']' then Break;
                Parser.CheckToken(',');
                Parser.NextToken;
              end;
          end;
        '(':
          begin
            Parser.NextToken;
            while Parser.Token <> ')' do ParseValue;
          end;
        '{':
          Parser.HexToBinary(Stream);
        '<':
          begin
            Parser.NextToken;
            while Parser.Token <> '>' do
            begin
              Parser.CheckTokenSymbol('item');
              Parser.NextToken;
              ParseTextOrderModifier(Parser);
              while not Parser.TokenSymbolIs('end') do ParseProperty(False);
              Parser.NextToken;
            end;
          end;
      else
        Parser.Error(SInvalidProperty);
      end;
      Parser.NextToken;
    end;
  end;

  procedure ParseProperty(IsForm: Boolean);
  var
    PropName: string;
    PropValue: Variant;
  begin
    Parser.CheckToken(toSymbol);
    PropName := Parser.TokenString;
    Parser.NextToken;
    while Parser.Token = '.' do
    begin
      Parser.NextToken;
      Parser.CheckToken(toSymbol);
      PropName := PropName + '.' + Parser.TokenString;
      Parser.NextToken;
    end;

    Parser.CheckToken('=');
    Parser.NextToken;
    PropValue := ParseValue;

    if IsForm then
    begin
      Inc(PropCount);
      if SameText(PropName, 'Left') then
        Info.Left := PropValue
      else if SameText(PropName, 'Top') then
        Info.Top := PropValue
      else if SameText(PropName, 'Width') or SameText(PropName, 'ClientWidth') then
        Info.Width := PropValue
      else if SameText(PropName, 'Height') or SameText(PropName, 'ClientHeight') then
        Info.Height := PropValue
      else if SameText(PropName, 'Caption') then
        Info.Caption := PropValue
      else
        Dec(PropCount);
    end;
  end;

  procedure ParseObject;
  var
    InheritedObject: Boolean;
    InlineObject: Boolean;
  begin
    InheritedObject := False;
    InlineObject := False;
    if Parser.TokenSymbolIs('INHERITED') then
    begin
      InheritedObject := True;
      Info.Kind := dkInherited;
    end
    else if Parser.TokenSymbolIs('INLINE') then
    begin
      InlineObject := True;
      Info.Kind := dkInline;
    end
    else
    begin
      Parser.CheckTokenSymbol('OBJECT');
      Info.Kind := dkObject;
    end;
    Parser.NextToken;
    ParseHeader(InheritedObject, InlineObject);
    while (PropCount < csPropCount) and
      not Parser.TokenSymbolIs('END') and
      not Parser.TokenSymbolIs('OBJECT') and
      not Parser.TokenSymbolIs('INHERITED') and
      not Parser.TokenSymbolIs('INLINE') do
      ParseProperty(True);
  end;

begin
  try
    Parser := TParser.Create(Stream);
    SaveSeparator := {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator;
    {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := '.';
    try
      PropCount := 0;
      ParseObject;
      Result := True;
    finally
      {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := SaveSeparator;
      Parser.Free;
    end;
  except
    Result := False;
  end;
end;

// �򵥽��������Ƹ�ʽ�� Dfm �õ� Info
function ParseBinaryDfmStream(Stream: TStream; Info: TDfmInfo): Boolean;
var
  SaveSeparator: Char;
  Reader: TReader;
  PropName: string;
  PropCount: Integer;

  procedure ParseHeader;
  var
    Flags: TFilerFlags;
    Position: Integer;
  begin
    Reader.ReadPrefix(Flags, Position);
    Info.FormClass := Reader.ReadStr;
    Info.Name := Reader.ReadStr;
    if Info.Name = '' then
      Info.Name := Info.FormClass;
  end;

  procedure ParseBinary;
  const
    BYTES_PER_LINE = 32;
  var
    I: Integer;
    Count: Longint;
    Buffer: array[0..BYTES_PER_LINE - 1] of Char;
  begin
    Reader.ReadValue;
    Reader.Read(Count, SizeOf(Count));
    while Count > 0 do
    begin
      if Count >= 32 then I := 32 else I := Count;
      Reader.Read(Buffer, I);
      Dec(Count, I);
    end;
  end;

  procedure ParseProperty(IsForm: Boolean); forward;

  function ParseValue: Variant;
  const
    LineLength = 64;
  var
    S: string;
  begin
    Result := Null;
    case Reader.NextValue of
      vaList:
        begin
          Reader.ReadValue;
          while not Reader.EndOfList do
            ParseValue;
          Reader.ReadListEnd;
        end;
      vaInt8, vaInt16, vaInt32:
        Result := Reader.ReadInteger;
      vaExtended:
        Result := Reader.ReadFloat;
      vaSingle:
        Result := Reader.ReadSingle;
      vaCurrency:
        Result := Reader.ReadCurrency;
      vaDate:
        Result := Reader.ReadDate;
      vaWString{$IFDEF COMPILER6_UP}, vaUTF8String{$ENDIF}:
        Result := Reader.ReadWideString;
      vaString, vaLString:
        Result := Reader.ReadString;
      vaIdent, vaFalse, vaTrue, vaNil, vaNull:
        Result := Reader.ReadIdent;
      vaBinary:
        ParseBinary;
      vaSet:
        begin
          Reader.ReadValue;
          while True do
          begin
            S := Reader.ReadStr;
            if S = '' then Break;
          end;
        end;
      vaCollection:
        begin
          Reader.ReadValue;
          while not Reader.EndOfList do
          begin
            if Reader.NextValue in [vaInt8, vaInt16, vaInt32] then
            begin
              ParseValue;
            end;
            Reader.CheckValue(vaList);
            while not Reader.EndOfList do ParseProperty(False);
            Reader.ReadListEnd;
          end;
          Reader.ReadListEnd;
        end;
      vaInt64:
      {$IFDEF COMPILER6_UP}
        Result := Reader.ReadInt64;
      {$ELSE}
        Result := Integer(Reader.ReadInt64);
      {$ENDIF}
    else
      raise EReadError.CreateResFmt(@sPropertyException,
        [Info.Name, DotSep, PropName, IntToStr(Ord(Reader.NextValue))]);
    end;
  end;

  procedure ParseProperty(IsForm: Boolean);
  var
    PropValue: Variant;
  begin
    PropName := Reader.ReadStr;
    PropValue := ParseValue;

    if IsForm then
    begin
      Inc(PropCount);
      if SameText(PropName, 'Left') then
        Info.Left := PropValue
      else if SameText(PropName, 'Top') then
        Info.Top := PropValue
      else if SameText(PropName, 'Width') then
        Info.Width := PropValue
      else if SameText(PropName, 'Height') then
        Info.Height := PropValue
      else if SameText(PropName, 'Caption') then
        Info.Caption := PropValue
      else
        Dec(PropCount);
    end;
  end;

  procedure ParseObject;
  begin
    ParseHeader;
    while (PropCount < csPropCount) and not Reader.EndOfList do
      ParseProperty(True);
  end;

begin
  try
    Reader := TReader.Create(Stream, 4096);
    SaveSeparator := {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator;
    {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := '.';
    try
      PropCount := 0;
      Reader.ReadSignature;
      ParseObject;
      Result := True;
    finally
      {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := SaveSeparator;
      Reader.Free;
    end;
  except
    Result := False;
  end;
end;

function ParseDfmStream(Stream: TStream; Info: TDfmInfo): Boolean;
var
  Pos: Integer;
  Signature: Integer;
  BOM: array[1..3] of AnsiChar;
begin
  Pos := Stream.Position;
  Signature := 0;
  Stream.Read(Signature, SizeOf(Signature));
  Stream.Position := Pos;
  if AnsiChar(Signature) in ['o','O','i','I',' ',#13,#11,#9] then
  begin
    Info.Format := dfText;
    Result := ParseTextDfmStream(Stream, Info);
  end
  else
  begin
    Pos := Stream.Position;
    Signature := 0;
    Stream.Read(BOM, SizeOf(BOM));
    Stream.Position := Pos;

    if ((BOM[1] = #$FF) and (BOM[2] = #$FE)) or // UTF8/UTF 16
      ((BOM[1] = #$EF) and (BOM[2] = #$BB) and (BOM[3] = #$BF)) then
    begin
      Info.Format := dfText;
      Result := ParseTextDfmStream(Stream, Info); // Only ANSI yet
    end
    else
    begin
      Stream.ReadResHeader;
      Pos := Stream.Position;
      Signature := 0;
      Stream.Read(Signature, SizeOf(Signature));
      Stream.Position := Pos;
      if Signature = Integer(FILER_SIGNATURE) then
      begin
        Info.Format := dfBinary;
        Result := ParseBinaryDfmStream(Stream, Info);
      end
      else
      begin
        Info.Format := dfUnknown;
        Result := False;
      end;
    end;
  end;
end;

function ParseDfmFile(const FileName: string; Info: TDfmInfo): Boolean;
var
  Stream: TFileStream;
begin
  try
    Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
    try
      Result := ParseDfmStream(Stream, Info);
    finally
      Stream.Free;
    end;
  except
    Result := False;
  end;
end;

function LoadTextDfmStreamToTree(Stream: TStream; Tree: TCnDfmTree): Boolean;
var
  SaveSeparator: Char;
  Parser: TParser;
  StartLeaf: TCnDfmLeaf;
begin
  Parser := TParser.Create(Stream);
  try
    SaveSeparator := {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator;
    {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := '.';
    try
      StartLeaf := Tree.AddChild(Tree.Root) as TCnDfmLeaf;
      ParseTextObjectToLeaf(Parser, Tree, StartLeaf as TCnDfmLeaf);
      Result := True;
    finally
      {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := SaveSeparator;
      Parser.Free;
    end;
  except
    Result := False;
  end;
end;

function LoadBinaryDfmStreamToTree(Stream: TStream; Tree: TCnDfmTree): Boolean;
var
  Reader: TReader;
  SaveSeparator: Char;
  StartLeaf: TCnDfmLeaf;
begin
  try
    Reader := TReader.Create(Stream, 4096);
    SaveSeparator := {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator;
    {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := '.';
    try
      Reader.ReadSignature;
      StartLeaf := Tree.AddChild(Tree.Root) as TCnDfmLeaf;
      ParseBinaryObjectToLeaf(Reader, StartLeaf);
      Result := True;
    finally
      {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := SaveSeparator;
      Reader.Free;
    end;
  except
    Result := False;
  end;
end;

function LoadMultiTextStreamToTree(Stream: TStream; Tree: TCnDfmTree): Boolean;
var
  Pos: Integer;
  Signature: Integer;
  SaveSeparator: Char;
  Parser: TParser;
  StartLeaf: TCnDfmLeaf;
begin
  Result := False;
  Pos := Stream.Position;
  Signature := 0;
  Stream.Read(Signature, SizeOf(Signature));
  Stream.Position := Pos;

  if AnsiChar(Signature) in ['o','O','i','I',' ',#13,#11,#9] then
  begin
    Tree.DfmFormat := dfText;
    SaveSeparator := {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator;
    {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := '.';

    Parser := TParser.Create(Stream);
    try
      while Parser.Token <> #0 do
      begin
        StartLeaf := Tree.AddChild(Tree.Root) as TCnDfmLeaf;
        try
          ParseTextObjectToLeaf(Parser, Tree, StartLeaf as TCnDfmLeaf);
        except
          // StartLeaf ����ʧ�ܣ����ܵ�β���ˣ�Ҫɾ��
          StartLeaf.Delete;
          Result := Tree.Count > 1;
          Exit;
        end;
      end;
      Result := True;
    finally
      {$IFDEF DELPHIXE3_UP}FormatSettings.{$ENDIF}DecimalSeparator := SaveSeparator;
      Parser.Free;
    end;
  end;
end;

function LoadDfmStreamToTree(Stream: TStream; Tree: TCnDfmTree): Boolean;
var
  Pos: Integer;
  Signature: Integer;
  BOM: array[1..3] of AnsiChar;
begin
  Result := False;
  Pos := Stream.Position;
  Signature := 0;
  Stream.Read(Signature, SizeOf(Signature));
  Stream.Position := Pos;

  if AnsiChar(Signature) in ['o','O','i','I',' ',#13,#11,#9] then
  begin
    Tree.DfmFormat := dfText;
    Result := LoadTextDfmStreamToTree(Stream, Tree);
  end
  else
  begin
    Pos := Stream.Position;
    Signature := 0;
    Stream.Read(BOM, SizeOf(BOM));
    Stream.Position := Pos;

    if ((BOM[1] = #$FF) and (BOM[2] = #$FE)) or // UTF8/UTF 16
      ((BOM[1] = #$EF) and (BOM[2] = #$BB) and (BOM[3] = #$BF)) then
    begin
      Tree.DfmFormat := dfText;
      Result := LoadTextDfmStreamToTree(Stream, Tree); // Only ANSI yet
    end
    else
    begin
      try
        Stream.ReadResHeader;
      except
        Exit; // ��������쳣���˳�
      end;

      Pos := Stream.Position;
      Signature := 0;
      Stream.Read(Signature, SizeOf(Signature));
      Stream.Position := Pos;
      if Signature = Integer(FILER_SIGNATURE) then
      begin
        Tree.DfmFormat := dfBinary;
        Result := LoadBinaryDfmStreamToTree(Stream, Tree);
      end
      else
      begin
        Tree.DfmFormat := dfUnknown;
        Result := False;
      end;
    end;
  end;
end;

function LoadDfmFileToTree(const FileName: string; Tree: TCnDfmTree): Boolean;
var
  Stream: TFileStream;
begin
  try
    Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
    try
      Result := LoadDfmStreamToTree(Stream, Tree);
    finally
      Stream.Free;
    end;
  except
    Result := False;
  end;
end;

function SaveTreeToStrings(const List: TStrings; Tree: TCnDfmTree): Boolean;
begin
  Result := Tree.SaveToStrings(List);
end;

function SaveTreeToDfmFile(const FileName: string; Tree: TCnDfmTree): Boolean;
var
  List: TStrings;
begin
  Result := False;
  if (FileName <> '') and (Tree <> nil) then
  begin
    List := TStringList.Create;
    try
      Result := Tree.SaveToStrings(List);
      List.SaveToFile(FileName);
    finally
      List.Free;
    end;
  end;
end;

function DecodeDfmStr(const QuotedStr: string): string;
var
  Stream: TMemoryStream;
  Parser: TParser;
  Reparse: Boolean;
{$IFDEF UNICODE}
  A: AnsiString;
{$ENDIF}
begin
  Result := QuotedStr;
  if QuotedStr = '' then
    Exit;

  Reparse := True;
  if Pos('#', Result) > 0 then
  begin
    Stream := nil;
    Parser := nil;

    try
      // ͨ�� Parser ����� #12345 ���� Unicode ת���ַ�
      Stream := TMemoryStream.Create;
  {$IFDEF UNICODE}  // Parser ֻ���� AnsiString ����
      A := AnsiString(QuotedStr);
      Stream.Write(A[1], Length(A));
  {$ELSE}
      Stream.Write(QuotedStr[1], Length(QuotedStr));
  {$ENDIF}
      Stream.Position := 0;

      Parser := TParser.Create(Stream);
      Parser.NextToken;

      Result := string(Parser.TokenWideString);

      Reparse := Result = ''; // # ��������ɹ����� Reparse ��Ϊ False�������ؽ���
    finally
      Parser.Free;
      Stream.Free;
    end;
  end;

  // ���Ͼ��� Parser�����ԭʼ������ #��������ͨ��������Ѿ�û������
  // �����ԭʼ����û #������������ַ������൱��ʧ���ˣ������ֹ�ȥ����
  if Reparse then
  begin
    Result := QuotedStr;
    if Length(Result) > 1 then
    begin
      if Result[1] = '''' then // ɾͷ����
        Delete(Result, 1, 1)
      else
        Exit;

      if Length(Result) > 0 then
      begin
        if Result[Length(Result)] = '''' then // ɾβ����
          Delete(Result, Length(Result), 1)
        else
          Exit;

        Result := StringReplace(Result, '''''', '''', [rfReplaceAll]); // ˫�����滻�ɵ�����
      end;
    end;
  end;
end;

{ TCnDfmTree }

constructor TCnDfmTree.Create;
begin
  inherited Create(TCnDfmLeaf);
end;

destructor TCnDfmTree.Destroy;
begin

  inherited;
end;

function TCnDfmTree.GetItems(AbsoluteIndex: Integer): TCnDfmLeaf;
begin
  Result := TCnDfmLeaf(inherited GetItems(AbsoluteIndex));
end;

function TCnDfmTree.GetRoot: TCnDfmLeaf;
begin
  Result := TCnDfmLeaf(inherited GetRoot);
end;

function TCnDfmTree.GetSameClassIndex(Leaf: TCnDfmLeaf): Integer;
var
  I: Integer;
begin
  Result := -1;
  if Leaf.Tree <> Self then
    Exit;

  for I := 0 to Count - 1 do
  begin
    if Items[I].ElementClass = Leaf.ElementClass then
      Inc(Result);
    if Items[I] = Leaf then
      Exit;
  end;
end;

procedure TCnDfmTree.SaveLeafToStrings(Leaf: TCnDfmLeaf; List: TStrings;
  Tab: Integer);
var
  I: Integer;
begin
  if (Leaf <> nil) and (Leaf.ElementClass <> '') then
  begin
    Leaf.AppendToStrings(List, Tab);
    for I := 0 to Leaf.Count - 1 do
      SaveLeafToStrings(Leaf.Items[I], List, Tab + csTabWidth);

    List.Append(Spc(Tab) + 'end');
  end;
end;

function TCnDfmTree.SaveToStrings(List: TStrings): Boolean;
begin
  Result := False;
  if List <> nil then
  begin
    List.Clear;
    // Root ������
    if Root.Count = 1 then
      SaveLeafToStrings(Root.Items[0], List, 0);
  end;
end;

{ TCnDfmLeaf }

procedure TCnDfmLeaf.AppendToStrings(List: TStrings; Tab: Integer);
var
  I, P: Integer;
  S, N, V: string;
begin
  if Tab < 0 then
    Tab := 0;

  if List <> nil then
  begin
    List.Add(Format('%s%s %s: %s', [Spc(Tab), LowerCase(SDfmKinds[FElementKind]), Text, FElementClass]));
    for I := 0 to FProperties.Count - 1 do
    begin
      S := FProperties[I];
      P := Pos(' = ', S);
      if P > 0 then
      begin
        N := Copy(S, 1, P - 1);
        V := Copy(S, P + 3, MaxInt);
        // �� V ��ÿ���س�������� Tab + csTabWidth ���ո���ʾ����
        V := StringReplace(V, #13#10, #13#10 + Spc(Tab + csTabWidth), [rfReplaceAll]);
        if (N <> '') and (V <> '') then
          List.Add(Format('%s%s = %s', [Spc(Tab + csTabWidth), N, V]));
      end;
    end;
  end;
end;

procedure TCnDfmLeaf.AssignTo(Dest: TPersistent);
var
  I: Integer;
  SourceStream, DestStream: TMemoryStream;
begin
  if Dest is TCnDfmLeaf then
  begin
    TCnDfmLeaf(Dest).ElementKind := FElementKind;
    TCnDfmLeaf(Dest).ElementClass := FElementClass;
    TCnDfmLeaf(Dest).Properties.Assign(FProperties);

    for I := 0 to FProperties.Count - 1 do
    begin
      SourceStream := TMemoryStream(FProperties.Objects[I]);
      if SourceStream <> nil then
      begin
        // ���ƶ������ڴ���
        DestStream := TMemoryStream.Create;
        DestStream.LoadFromStream(SourceStream);
        TCnDfmLeaf(Dest).Properties.Objects[I] := DestStream;
      end;
    end;
  end;
  inherited;
end;

constructor TCnDfmLeaf.Create(ATree: TCnTree);
begin
  inherited;
  FProperties := TStringList.Create;
end;

destructor TCnDfmLeaf.Destroy;
var
  I: Integer;
begin
  for I := 0 to FProperties.Count - 1 do
    if FProperties.Objects[I] <> nil then
      FProperties.Objects[I].Free;
  FProperties.Free;
  inherited;
end;

function TCnDfmLeaf.GetItems(Index: Integer): TCnDfmLeaf;
begin
  Result := TCnDfmLeaf(inherited GetItems(Index));
end;

function TCnDfmLeaf.GetPropertyValue(const PropertyName: string): string;
var
  I, D: Integer;
begin
  Result := '';
  for I := 0 to FProperties.Count - 1 do
  begin
    D := Pos('=', FProperties[I]);
    if D > 1 then
    begin
      if PropertyName = Trim(Copy(FProperties[I], 1, D - 1)) then
      begin
        Result := Trim(Copy(FProperties[I], D + 1, MaxInt));
        Exit;
      end;
    end;
  end;
end;

function TCnDfmLeaf.GetTree: TCnDfmTree;
begin
  Result := TCnDfmTree(inherited Tree);
end;

procedure TCnDfmLeaf.SetItems(Index: Integer; const Value: TCnDfmLeaf);
begin
  inherited SetItems(Index, Value);
end;

end.
