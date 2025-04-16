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

unit CnGetThread;
{ |<PRE>
================================================================================
* ������ƣ�CnDebugViewer
* ��Ԫ���ƣ���ȡ�̵߳�Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2018.12.17
*               ������̹��˹��� (by zjy@cnpack.org)
*           2006.10.15
*               �����ȡ OutputDebugString ���ݵĹ���
*           2005.01.01
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

uses
  Classes, SysUtils, Windows, Forms, Contnrs,
  CnViewCore, CnDebugIntf, CnMsgClasses;

type
  TProcessFilter = class
  private
    FWhiteList: TList;
    FBlackList: TList;
    FWhiteCache: string;
    FBlackCache: string;
    FChangeCountCache: Integer;
  public
    constructor Create;
    destructor Destroy; override;
    function FilterProcessId(ProcId: Integer): Boolean;
  end;

  TGetDebugThread = class(TThread)
  {* ��ȡ CnDebugger ���ݵ��߳�}
  private
    FCount: Cardinal;
    FPaused: Boolean;
    FFilter: TProcessFilter;
  protected
    procedure AddADescToStore(var ADesc: TCnMsgDesc);
    procedure Execute; override;
  public
    constructor Create(CreateSuspended: Boolean);
    destructor Destroy; override;
    property Paused: Boolean read FPaused write FPaused;
  end;

  TDbgGetDebugThread = class(TGetDebugThread)
  {* ��ȡ OutputDebugString ���ݵ��߳�}
  protected
    procedure Execute; override;
  end;

implementation

uses
  CnMdiView;

const
  MAX_WAIT_COUNT = 10;

{ TProcessFilter }

constructor TProcessFilter.Create;
begin
  FWhiteList := TList.Create;
  FBlackList := TList.Create;
  FChangeCountCache := -1;
end;

destructor TProcessFilter.Destroy;
begin
  FWhiteList.Free;
  FBlackList.Free;
  inherited;
end;

function TProcessFilter.FilterProcessId(ProcId: Integer): Boolean;
var
  ProcName: string;

  function ListToStr(Lines: string): string;
  var
    i: Integer;
    AList: TStringList;
    S: string;
  begin
    AList := TStringList.Create;
    try
      AList.CommaText := Lines;
      if AList.Count > 0 then
      begin
        Result := '\';
        for i := 0 to AList.Count - 1 do
        begin
          S := LowerCase(AList[i]);
          if ExtractFileExt(S) = '' then
            S := S + '.exe';
          Result := Result + S + '\';
        end;
      end
      else
        Result := '';
    finally
      AList.Free;
    end;
  end;
begin
  if (CnViewerOptions <> nil) and (CnViewerOptions.ChangeCount <> FChangeCountCache) then
  begin
    FWhiteList.Clear;
    FBlackList.Clear;
    FWhiteCache := ListToStr(CnViewerOptions.WhiteList);
    FBlackCache := ListToStr(CnViewerOptions.BlackList);
    FChangeCountCache := CnViewerOptions.ChangeCount;
  end;

  Result := True;
  if CnViewerOptions.UseBlackList then // ��ʾ��������������н���
  begin
    if FBlackList.IndexOf(Pointer(ProcId)) >= 0 then
      Result := True
    else if FBlackCache <> '' then
    begin
      ProcName := '\' + LowerCase(ExtractFileName(GetProcNameFromProcessID(ProcId))) + '\';
      Result := Pos(ProcName, FBlackCache) = 0;
      if Result then
        FBlackList.Add(Pointer(ProcId));
    end;
  end
  else // ֻ��ʾ���������ڵĽ���
  begin
    if FWhiteList.IndexOf(Pointer(ProcId)) >= 0 then
      Result := True
    else if FWhiteCache <> '' then
    begin
      ProcName := '\' + LowerCase(ExtractFileName(GetProcNameFromProcessID(ProcId))) + '\';
      Result := Pos(ProcName, FWhiteCache) > 0;
      if Result then
        FWhiteList.Add(Pointer(ProcId));
    end;
  end;
end;

{ GetDebug }

procedure TGetDebugThread.AddADescToStore(var ADesc: TCnMsgDesc);
var
  AStore: TCnMsgStore;
  StoreInited: Boolean;
  VarName, VarValue: string;
  AMsg: array [0..CnMaxMsgLength] of AnsiChar;
  Size, SplitterIdx: Integer;
begin
  if not FFilter.FilterProcessId(ADesc.Annex.ProcessId) then
    Exit;
  
  if ADesc.Annex.MsgType in [Ord(cmtWatch), Ord(cmtClearWatch)] then
  begin
    FillChar(AMsg, SizeOf(AMsg), 0);
    Size := ADesc.Length - SizeOf(ADesc.Annex) - SizeOf(DWord);
    Move(ADesc.Msg[0], AMsg, Size);

    SplitterIdx := Pos('|', AMsg);
    if SplitterIdx > 1 then // Clear ʱû����
      VarName := Copy(AMsg, 1, SplitterIdx - 1)
    else
      VarName := AMsg;

    if VarName <> '' then
    begin
      if (ADesc.Annex.MsgType = Ord(cmtWatch)) and (SplitterIdx > 1) then
      begin
        VarValue := Copy(AMsg, SplitterIdx + 1, MaxInt);
        CnMsgManager.PutWatch(VarName, VarValue);
      end
      else
        CnMsgManager.ClearWatch(VarName);

      CnMsgManager.DoWatchChanged;
      Exit;
    end;
  end;

  AStore := CnMsgManager.IndexOf(ADesc.Annex.ProcessId);
  StoreInited := False;
  if AStore = nil then
  begin
    AStore := CnMsgManager.IndexOf(0);
    if AStore = nil then
    begin
      if Application.MainForm <> nil then
        if not (csDestroying in Application.MainForm.ComponentState) then
        begin
          AStore := CnMsgManager.AddStore(0, SCnNoneProcName);
          AStore.ProcessID := ADesc.Annex.ProcessId;
          AStore.ProcName := GetProcNameFromProcessID(AStore.ProcessID);
{$IFDEF WIN64}
          PostMessage(Application.MainForm.Handle, WM_USER_NEW_FORM, NativeInt(AStore), 0);
{$ELSE}
          PostMessage(Application.MainForm.Handle, WM_USER_NEW_FORM, Integer(AStore), 0);
{$ENDIF}
        end;
    end;

    if not StoreInited and (AStore <> nil) then
    begin
      AStore.ProcessID := ADesc.Annex.ProcessId;
      AStore.ProcName := GetProcNameFromProcessID(AStore.ProcessID);
    end;
  end;

  // ���޿���Ļ��Ӧ�� Store�������
  if AStore <> nil then
  begin
    // ���������ʱ�ȿ�ʼ������ȡ���ȴ����ݶ���ʱ�ٸ��½���
    AStore.BeginUpdate;
    AStore.AddMsgDesc(@ADesc);
  end;
end;

constructor TGetDebugThread.Create(CreateSuspended: Boolean);
begin
  FFilter := TProcessFilter.Create;
  inherited Create(CreateSuspended);
end;

destructor TGetDebugThread.Destroy;
begin
  FFilter.Free;
  inherited;
end;

procedure TGetDebugThread.Execute;
var
  Len, RestLen, QueueSize: Integer;
  Desc: PCnMsgDesc;
  ADesc: TCnMsgDesc;
  Front, Tail: Integer;
  Res: DWORD;
  QueueAlreadyEmpty: Boolean;

  procedure CheckExit;
  var
    Count: Integer;
  begin
    if HMutex <> 0 then
    begin
      Count := 0;
      repeat
        Res := WaitForSingleObject(HMutex, CnWaitMutexTime);
        if Count > 0 then
          Sleep(0);
        Inc(Count);
      until (Res = WAIT_OBJECT_0) or (Count = MAX_WAIT_COUNT);

      CloseHandle(HMutex);
      HMutex := 0;
    end;
  end;

begin
  DebugDebuggerLog('TGetDebugThread Start');

  PostStartEvent;
  QueueSize := CnMapSize - CnHeadSize;
  QueueAlreadyEmpty := False;

  if HMutex = 0 then
    HMutex := CreateMutex(nil, False, PChar(SCnDebugQueueMutexName));

  while not Terminated do
  begin
    Res := WaitForSingleObject(HEvent, CnWaitEventTime);
    if Res = WAIT_FAILED then // ��ʹ��ʱҲ���ж�һ�¶���״̬
    begin
      Sleep(0);
      Continue;
    end;

    if not QueueAlreadyEmpty and (PHeader^.QueueFront = PHeader^.QueueTail) then
    begin
      // ���иտգ��ɿ�ʼ���½���
      QueueAlreadyEmpty := True;
      if (Application.MainForm <> nil) and not (csDestroying in Application.MainForm.ComponentState) then
        PostMessage(Application.MainForm.Handle, WM_USER_UPDATE_STORE, 0, 0);
      Sleep(0);
      Continue;
    end
    else
      QueueAlreadyEmpty := False;

    Res := WaitForSingleObject(HMutex, CnWaitMutexTime);
    if (Res = WAIT_FAILED) or (Res = WAIT_TIMEOUT) then
    begin
      // Sleep(0);
      Continue;
    end;

    Front := PHeader^.QueueFront;
    Tail := PHeader^.QueueTail;
    if Front = Tail then
    begin
      if Terminated then
      begin
        CheckExit;
        DebugDebuggerLog('Front = Tail and Terminated');
        Exit;
      end;
      ReleaseMutex(HMutex);
      Continue;
    end;

{$IFDEF WIN64}
    Desc := PCnMsgDesc(Front + PHeader^.DataOffset + NativeInt(PBase));
{$ELSE}
    Desc := PCnMsgDesc(Front + PHeader^.DataOffset + Integer(PBase));
{$ENDIF}

    if not Paused then
    begin
      FillChar(ADesc, SizeOf(ADesc), 0);
      Len := Desc^.Length;

      if Len = 0 then // ���ⷢ�Ͷ˳�������ѭ��
      begin
        PHeader^.QueueFront := 0;
        PHeader^.QueueTail := 0;
        if Terminated then
        begin
          CheckExit;
          DebugDebuggerLog('NOT Pause and Len = 0 and Terminated');
          Exit;
        end;
        ReleaseMutex(HMutex);
        Continue;
      end;
      
      if Front + Len < QueueSize then
        Move(Desc^, ADesc, Len)
      else
      begin
        RestLen := QueueSize - Front;
        Move(Desc^, ADesc, RestLen);
{$IFDEF WIN64}
        Move(Pointer(PHeader^.DataOffset + NativeInt(PBase))^,
          Pointer(NativeInt(@ADesc) + RestLen)^, Len - RestLen);
{$ELSE}
        Move(Pointer(PHeader^.DataOffset + Integer(PBase))^,
          Pointer(Integer(@ADesc) + RestLen)^, Len - RestLen);
{$ENDIF}
      end;

      EnterCriticalSection(CSMsgStore);
      try
        AddADescToStore(ADesc);
        Inc(FCount);
      finally
        LeaveCriticalSection(CSMsgStore);
      end;
    end; // ��ͣʱ���ָ��Ȼ���ָ�룬��ȡ������

    Inc(PHeader^.QueueFront, Desc^.Length);
    if PHeader^.QueueFront >= QueueSize then
      PHeader^.QueueFront := PHeader^.QueueFront mod QueueSize;

    if Terminated then
    begin
      CheckExit;
      DebugDebuggerLog('Check and Terminated');
      Exit;
    end;

    ReleaseMutex(HMutex);
    if HFlush = 0 then
      HFlush := OpenEvent(EVENT_MODIFY_STATE, False, PChar(SCnDebugFlushEventName));
    if HFlush <> 0 then
      SetEvent(hFlush);
  end;

  if Terminated then
  begin
    CheckExit;
    DebugDebuggerLog('Loop out and Terminated');
    Exit;
  end;
end;

{ TDbgGetDebugThread }

procedure TDbgGetDebugThread.Execute;
var
  Res: DWORD;
  ADesc: TCnMsgDesc;
  PPid: PDWORD;
  PStr: PChar;
  Len: Integer;
begin
  if not SysDebugReady then
    InitSysDebug;

  while not Terminated do
  begin
    if not SysDebugReady then
    begin
      Sleep(0);
      Continue;
    end;

    if not SetEvent(HSysBufferReady) then
    begin
      Sleep(0);
      Continue;
    end;

    Res := WaitForSingleObject(HSysDataReady, CnWaitEventTime);
    if Res <> WAIT_OBJECT_0 then
    begin
      Sleep(0);
      Continue;
    end;

    if Paused then
    begin
      Sleep(0);
      Continue;
    end;

    FillChar(ADesc, SizeOf(ADesc), 0);
    PPid := PDWORD(PSysDbgBase);
{$IFDEF WIN64}
    PStr := PChar(NativeInt(PSysDbgBase) + SizeOf(DWORD));
{$ELSE}
    PStr := PChar(Integer(PSysDbgBase) + SizeOf(DWORD));
{$ENDIF}

    ADesc.Annex.ProcessId := PPid^;
    
    // OutputDebugString �޶�Ӧ��Ϣ�������Ҫ�ֹ���д
    ADesc.Annex.Level := CnDefLevel;
    ADesc.Annex.MsgType := Ord(cmtSystem);
    // �� ThreadId���� Tag

    // �޷��Ͷ�ʱ��������Ʋ��ý��ն�ʱ���
    ADesc.Annex.TimeStampType := Ord(ttDateTime);
    ADesc.Annex.MsgDateTime := Date + Time;
    Len := StrLen(PStr);
    if Len >= DbWinBufferSize - SizeOf(DWORD) then
      Len := DbWinBufferSize - SizeOf(DWORD);
    Move(PStr^, ADesc.Msg[0], Len);

    ADesc.Length := Len + SizeOf(TCnMsgAnnex) + SizeOf(Integer) + 1;

    EnterCriticalSection(CSMsgStore);
    try
      AddADescToStore(ADesc);
    finally
      LeaveCriticalSection(CSMsgStore);
    end;

    // ���½���
    if (Application.MainForm <> nil) and not (csDestroying in Application.MainForm.ComponentState) then
      PostMessage(Application.MainForm.Handle, WM_USER_UPDATE_STORE, 0, 0);
  end;
end;

end.
