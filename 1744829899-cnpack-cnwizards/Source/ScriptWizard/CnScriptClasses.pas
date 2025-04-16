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

unit CnScriptClasses;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ��ű���չ�൥Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWinXP SP2 + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����ô����е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2006.09.20 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNSCRIPTWIZARD}

{$IFDEF SUPPORT_PASCAL_SCRIPT}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  CnCommon, uPSComponent, uPSCompiler, uPSRuntime;

type
  TCnPSPlugin = class(TPSPlugin)
  public
    procedure CompOnUses1(CompExec: TPSScript); virtual;
    {* ���û�ʵ��ʹ�� uses �õ�Ԫʱ��ע�� }
  end;

  TCnPSScript = class(TPSScript)
  protected
    function DoOnUnknowUses(Sender: TPSPascalCompiler; const Name: AnsiString):
      Boolean; override;
  public
    destructor Destroy; override;
  end;

  TCnExecResult = (erSucc, erCompileError, erExecError);
  
  TCnReadlnEvent = procedure (const Prompt: string; var Text: string) of object;
  TCnWritelnEvent = procedure (const Text: string) of object;

{$IFDEF DELPHI2009_UP}
  TPSOnCompImport = TPSOnCompImportEvent;
  TPSOnExecImport = TPSOnExecImportEvent;
{$ENDIF}

  TCnScriptExec = class
  private
    PSScript: TPSScript;
    FOnCompile: TPSEvent;
    FOnExecute: TPSEvent;
    FOnCompImport: TPSOnCompImport;
    FOnExecImport: TPSOnExecImport;
    FSearchPath: TStrings;
    FScripFile: string;
    FOnReadln: TCnReadlnEvent;
    FOnWriteln: TCnWritelnEvent;
    function PSScriptNeedFile(Sender: TObject; const OrginFileName: AnsiString;
      var FileName, Output: AnsiString): Boolean;
    procedure PSScriptCompImport(Sender: TObject; X: TIFPSPascalcompiler);
    procedure PSScriptExecute(Sender: TPSScript);
    procedure PSScriptExecImport(Sender: TObject; Exec: TIFPSExec;
      X: TIFPSRuntimeClassImporter);
    procedure PSScriptCompile(Sender: TPSScript);
  public
    constructor Create;
    destructor Destroy; override;

    function ExecScript(Script: string; var Msg: string): TCnExecResult;
    function CompileScript(Script: string; var Msg: string): TCnExecResult;

    function FindFileInSearchPath(const OrgName, FileName: string;
      var OutName: string): Boolean;

    property ScripFile: string read FScripFile write FScripFile;
    property SearchPath: TStrings read FSearchPath;
    property Engine: TPSScript read PSScript;
    property OnCompile: TPSEvent read FOnCompile write FOnCompile;
    property OnExecute: TPSEvent read FOnExecute write FOnExecute;
    property OnCompImport: TPSOnCompImport read FOnCompImport write FOnCompImport;
    property OnExecImport: TPSOnExecImport read FOnExecImport write FOnExecImport;
    property OnReadln: TCnReadlnEvent read FOnReadln write FOnReadln;
    property OnWriteln: TCnWritelnEvent read FOnWriteln write FOnWriteln;
  end;

  TPSPluginClass = class of TPSPlugin;

function RegisterCnScriptPlugin(APluginClass: TPSPluginClass): Integer;
{* ע��һ���ű������ }

{$ENDIF}

{$ENDIF CNWIZARDS_CNSCRIPTWIZARD}

implementation

{$IFDEF CNWIZARDS_CNSCRIPTWIZARD}

{$IFDEF SUPPORT_PASCAL_SCRIPT}

{ TCnScriptExec }

var
  FPluginClasses: TList;

// ע��һ���ű������
function RegisterCnScriptPlugin(APluginClass: TPSPluginClass): Integer;
begin
  if FPluginClasses = nil then
    FPluginClasses := TList.Create;
  Result := FPluginClasses.Add(APluginClass);
end;

{ TCnPSPlugin }

procedure TCnPSPlugin.CompOnUses1(CompExec: TPSScript);
begin

end;

{ TCnPSScript }

function TCnPSScript.DoOnUnknowUses(Sender: TPSPascalCompiler;
  const Name: AnsiString): Boolean;
var
  I: Integer;
  Plugin: TPSPlugin;
  CName: string;
begin
  for I := 0 to Plugins.Count - 1 do
  begin
    Plugin := TPSPluginItem(Plugins.Items[I]).Plugin;
    CName := Plugin.ClassName;
    if Pos('_', CName) > 0 then
      CName := Copy(CName, Pos('_', CName) + 1, MaxInt);
    if SameText(CName, string(Name)) then
    begin
      // ֻ������ʱע��ĵ�Ԫ
      if Plugin is TCnPSPlugin then
        TCnPSPlugin(Plugin).CompOnUses1(Self);
      Result := True;
      Exit;
    end;
  end;
  Result := False;
end;

destructor TCnPSScript.Destroy;
var
  I: Integer;
begin
  // ��ǰ�ͷŲ�����Ա�������ͷ�ʱ����
  for I := Plugins.Count - 1 downto 0 do
    TPSPluginItem(Plugins.Items[I]).Plugin.Free;
  inherited Destroy;
end;

function ScriptFileName(Caller: TPSExec; p: TPSExternalProcRec;
  Global, Stack: TPSStack): Boolean;
begin
  Stack.SetString(-1, TCnScriptExec(p.Ext1).ScripFile);
  Result := True;
end;

function _Readln(Caller: TPSExec; p: TPSExternalProcRec;
  Global, Stack: TPSStack): Boolean;
var
  S: string;
begin
  if Assigned(TCnScriptExec(p.Ext1).OnReadln) then
    TCnScriptExec(p.Ext1).OnReadln(Stack.GetString(-2), S);
  Stack.SetString(-1, S);
  Result := True;
end;

function _Writeln(Caller: TPSExec; p: TPSExternalProcRec;
  Global, Stack: TPSStack): Boolean;
begin
  if Assigned(TCnScriptExec(p.Ext1).OnWriteln) then
    TCnScriptExec(p.Ext1).OnWriteln(Stack.GetString(-1));
  Result := True;
end;

{ TCnScriptExec }

constructor TCnScriptExec.Create;
var
 I: Integer;
begin
  FSearchPath := TStringList.Create;
  PSScript := TCnPSScript.Create(nil);
  PSScript.UsePreProcessor := True;
  PSScript.OnNeedFile := PSScriptNeedFile;
  PSScript.OnCompImport := PSScriptCompImport;
  PSScript.OnExecImport := PSScriptExecImport;
  PSScript.OnCompile := PSScriptCompile;
  PSScript.OnExecute := PSScriptExecute;

  if FPluginClasses <> nil then
  begin
    for I := 0 to FPluginClasses.Count - 1 do
      TPSPluginItem(PSScript.Plugins.Add).Plugin := TPSPluginClass(FPluginClasses[I]).Create(PSScript);
  end;
end;

destructor TCnScriptExec.Destroy;
begin
  FSearchPath.Free;
  PSScript.Free;
  inherited;
end;

function TCnScriptExec.FindFileInSearchPath(const OrgName, FileName: string;
  var OutName: string): Boolean;

  function LinkPath(const Head, Tail: string): string;
  var
    AHead, ATail: string;
    I: Integer;
  begin
    if Head = '' then
    begin
      Result := Tail;
      Exit;
    end;

    if Tail = '' then
    begin
      Result := Head;
      Exit;
    end;

    AHead := StringReplace(Head, '/', '\', [rfReplaceAll]);
    ATail := StringReplace(Tail, '/', '\', [rfReplaceAll]);
    if Copy(ATail, 1, 2) = '.\' then
      Delete(ATail, 1, 2);
      
    if AHead[Length(AHead)] = '\' then
      Delete(AHead, Length(AHead), MaxInt);
    I := Pos('..\', ATail);
    while I > 0 do
    begin
      AHead := _CnExtractFileDir(AHead);
      Delete(ATail, 1, 3);
      I := Pos('..\', ATail);
    end;
    
    Result := AHead + '\' + ATail;
  end;
var
  I: Integer;
begin
  Result := True;

  OutName := LinkPath(_CnExtractFileDir(OrgName), FileName);
  if FileExists(OutName) then
    Exit;

  OutName := LinkPath(_CnExtractFileDir(ScripFile), FileName);
  if FileExists(OutName) then
    Exit;

  for I := 0 to FSearchPath.Count - 1 do
  begin
    OutName := LinkPath(FSearchPath[I], FileName);
    if FileExists(OutName) then
      Exit;
  end;

  OutName := FileName;
  if FileExists(OutName) then
    Exit;
    
  Result := False;
  OutName := '';
end;

function TCnScriptExec.PSScriptNeedFile(Sender: TObject;
  const OrginFileName: AnsiString; var FileName, Output: AnsiString): Boolean;
var
  FullFile: string;
begin
  if FindFileInSearchPath(string(OrginFileName), string(FileName), FullFile) and
    FileExists(FullFile) then
  begin
    with TStringList.Create do
    try
      LoadFromFile(FullFile);
      Output := AnsiString(Text);
    finally
      Free;
    end;
    Result := True;
  end
  else
    Result := False;
end;

procedure TCnScriptExec.PSScriptCompImport(Sender: TObject;
  X: TIFPSPascalcompiler);
begin
  X.AddFunction('function ScriptFileName: string;');
  X.AddFunction('function Readln(const Msg: string): string;');
  X.AddFunction('procedure Writeln(const Text: string);');
  if Assigned(FOnCompImport) then
    FOnCompImport(Sender, X);
end;

procedure TCnScriptExec.PSScriptExecImport(Sender: TObject; Exec: TIFPSExec;
  X: TIFPSRuntimeClassImporter);
begin
  Exec.RegisterFunctionName('ScriptFileName', ScriptFileName, Self, nil);
  Exec.RegisterFunctionName('Readln', _Readln, Self, nil);
  Exec.RegisterFunctionName('Writeln', _Writeln, Self, nil);
  if Assigned(FOnExecImport) then
    FOnExecImport(Sender, Exec, X);
end;

procedure TCnScriptExec.PSScriptCompile(Sender: TPSScript);
begin
  if Assigned(FOnCompile) then
    FOnCompile(Sender);
end;

procedure TCnScriptExec.PSScriptExecute(Sender: TPSScript);
begin
  if Assigned(FOnExecute) then
    FOnExecute(Sender);
end;

function TCnScriptExec.CompileScript(Script: string;
  var Msg: string): TCnExecResult;
var
  I: Integer;
begin
  PSScript.Script.Text := Script;
  if PSScript.Compile then
    Result := erSucc
  else
  begin
    for I := 0 to PSScript.CompilerMessageCount - 1 do
      Msg := Msg + string(PSScript.CompilerErrorToStr(I)) + #13#10;
    Result := erCompileError;
  end;
end;

function TCnScriptExec.ExecScript(Script: string; var Msg: string): TCnExecResult;
var
  I: Integer;
begin
  PSScript.Script.Text := Script;
  if PSScript.Compile then
  begin
    if PSScript.Execute then
      Result := erSucc
    else
    begin
      Msg := string(PSScript.ExecErrorToString);
      Result := erExecError;
    end;
  end
  else
  begin
    for I := 0 to PSScript.CompilerMessageCount - 1 do
      Msg := Msg + string(PSScript.CompilerErrorToStr(I)) + #13#10;
    Result := erCompileError;
  end;
end;

initialization

finalization
  if FPluginClasses <> nil then
    FPluginClasses.Free;

{$ENDIF}

{$ENDIF SUPPORT_PASCAL_SCRIPT}
end.
