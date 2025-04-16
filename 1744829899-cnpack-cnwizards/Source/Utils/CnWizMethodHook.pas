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

unit CnWizMethodHook;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ����󷽷��ҽӵ�Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע���õ�Ԫ�����ҽ� IDE �ڲ���ķ���
*           32 λ��ͳһʹ�������תҲ�� E9 �� 32 λƫ�ƣ�һ��ûɶ���⡣
*           64 λ�����Ҳ�� E9 �� 32 λƫ�ƣ���ô DLL ���ڴ�ռ���̫Զ�ͻ�������ȥ
*           64 λ���� 25FF �� RIP ƫ�ƴ��� 8 �ֽ���Ϊ������ת��ַ��ģʽ��BPL ����ˣ���
*           ��ͬ�����ڸ� 8 �ֽڴ洢��λ������ҽӵķ���̫Զ�����⡣
*
*           �������ѡ��һ���� 64 λ��ʹ�� DDetours ����ܡ����ǿ������֣���ʵ�֣���
*              push address.low32
*              mov dword [rsp+4], address.high32
*              ret
*           �ô����ܸ������пռ䲻�õ���̫Զ�������� 14 ���ֽڣ�Զ�� 32 λ�µ� 5 ����
*
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2025.02.09
*               64 λ�����Ƴ�����ת�� Hook
*           2025.02.07
*               ���� GetBplMethodAddress �� 64 λ�µĴ��󣬵���ת����������
*               ��ǿ���� 64 λ��ʹ�� DDetours
*           2024.02.04
*               �� 64 λ֧�ִ� CnMethodHook �����롣���������Ƿ�ʹ�� DDetours
*           2018.01.12
*               �����ʼ��ʱ���Զ��ҽӵĿ��ƣ�����ӿں�������ʵ��ַ��ȡ
*           2014.10.01
*               �� DDetours ���ø�Ϊ��̬
*           2014.08.28
*               ���� DDetours ʵ�ֵ���
*           2003.10.27
*               ʵ�����Ա༭�������ҽӺ��ļ���
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, SysUtils, Classes, CnNative {$IFDEF USE_DDETOURS_HOOK}, DDetours{$ENDIF};

type
  PCnLongJump = ^TCnLongJump;
  TCnLongJump = packed record
    JmpOp: Byte;        // Jmp �����תָ�Ϊ $E9��32 λ�� 64 λͨ��
{$IFDEF CPU64BITS}
    Addr: DWORD;        // 64 λ�µ���ת������Ե�ַ��Ҳ�� 32 λ�������ǲ��� DLL ����̫Զ�����
{$ELSE}
    Addr: Pointer;      // ��ת���� 32 λ��Ե�ַ
{$ENDIF}
  end;

{$IFDEF CPU64BITS}

  { 64 λ�µı��೤��ת��࣬ռ 14 �ֽ�
      PUSH addr.low32
      MOV DWORD [rsp+4], addr.high32
      RET
    Ҳ�� 68 44332211
         C74424 04 88776655
         C3
    ����ת�� $5566778811223344
  }
  PCnLongJump64 = ^TCnLongJump64;
  TCnLongJump64 = packed record
    PushOp: Byte;         // $68
    AddrLow32: DWORD;
    MovOp: DWORD;         // $042444C7
    AddrHigh32: DWORD;
    RetOp: Byte;          // $C3
  end;

{$ENDIF}

  TCnMethodHook = class
  {* ��̬�� dynamic �����ҽ��࣬���ڹҽ����о�̬����������Ϊ dynamic �Ķ�̬������
     ����ͨ���޸�ԭ�������ǰ 5/14 �ֽڣ���Ϊ��תָ����ʵ�ַ����ҽӲ�������ʹ��ʱ
     �뱣֤ԭ������ִ���������� 5/14 �ֽڣ�������ܻ�������غ����}
  private
    FUseDDteours: Boolean;
    FHooked: Boolean;
    FOldMethod: Pointer;
    FNewMethod: Pointer;
    FTrampoline: Pointer;
    FSaveData: TCnLongJump;
{$IFDEF CPU64BITS}
    FSaveData64: TCnLongJump64; // 64 λԶ���ı�������
    FFar: Boolean;              // 64 λ���Ƿ�̫ԶҪ�ø�������ת
    procedure InitLongJump64(JmpPtr: PCnLongJump64);
{$ENDIF}
  public
    constructor Create(const AOldMethod, ANewMethod: Pointer; UseDDteoursHook: Boolean = False;
      DefaultHook: Boolean = True);
    {* ������������Ϊԭ������ַ���·�����ַ��ע�������ר�Ұ���ʹ�ã�ԭ������ַ
       ���� GetBplMethodAddress ת������ʵ��ַ�����������ú���Զ��ҽӴ���ķ�����
     |<PRE>
       �������Ҫ�ҽ� TTest.Abc(const A: Integer) ���������Զ����·���Ϊ��
       procedure MyAbc(ASelf: TTest; const A: Integer);
       �˴� MyAbc Ϊ��ͨ���̣���Ϊ������������һ������Ϊ Self���ʴ˴�����һ��
       ASelf: TTest ������֮��ԣ�ʵ�ִ����п��԰�����������ʵ�������ʡ�
     |</PRE>}
    destructor Destroy; override;
    {* ����������ȡ���ҽ�}

    property Hooked: Boolean read FHooked;
    {* �Ƿ��ѹҽ�}
    procedure HookMethod; virtual;
    {* ���¹ҽӣ������Ҫִ��ԭ���̣���ʹ���� UnhookMethod������ִ����ɺ����¹ҽ�}
    procedure UnhookMethod; virtual;
    {* ȡ���ҽӣ������Ҫִ��ԭ���̣�����ʹ�� UnhookMethod���ٵ���ԭ���̣���������}
    property Trampoline: Pointer read FTrampoline;
    {* DDetours �ҽӺ�ľɷ�����ַ������粻�л��ҽ�״̬��ֱ�ӵ��á�
       �粻ʹ�� DDetours����Ϊ nil}
    property UseDDteours: Boolean read FUseDDteours;
    {* �Ƿ�ʹ�� UseDDteours ����йҽ�}
  end;

function GetBplMethodAddress(Method: Pointer): Pointer;
{* ������ BPL ��ʵ�ʵķ�����ַ����ר�Ұ����� @TPersistent.Assign ���ص���ʵ��
   һ�� Jmp ��ת��ַ���ú������Է����� BPL �з�������ʵ��ַ��֧�� 32 λ�� 64 λ
   ���� 64 λ��Ŀǰֻ��������ת�� JMP QWORD PTR [RIP + offset] Ҳ�� $25FF ������}

function GetInterfaceMethodAddress(const AIntf: IUnknown;
  MethodIndex: Integer): Pointer;
{* ���� Delphi ��֧�� @AIntf.Proc �ķ�ʽ���ؽӿڵĺ�����ڵ�ַ������ Self ָ��Ҳ
   ��ƫ�����⡣���������ڷ��� AIntf �ĵ� MethodIndex ����������ڵ�ַ����������
   Self ָ���ƫ�����⡣
   MethodIndex �� 0 ��ʼ��0��1��2 �ֱ���� QueryInterface��_AddRef��_Release��
   ע�� MethodIndex �����߽��飬�����˸� Interface �ķ����������}

implementation

resourcestring
  SCnMemoryWriteError = 'Error Writing Method Memory (%s).';
  SCnFailInstallHook = 'Failed to Install Method Hook';
  SCnFailUninstallHook = 'Failed to Uninstall Method Hook';
  SCnErrorNoDDetours = 'DDetours NOT Included. Can NOT Hook.';

const
  csJmpCode = $E9;              // �����תָ�������
  csJmp32Code = $25FF;          // BPL ����ڵ���ת�����룬32 λ�� 64 λͨ��

type
{$IFDEF CPU64BITS}
  TCnAddressInt = NativeInt;
{$ELSE}
  TCnAddressInt = Integer;
{$ENDIF}

var
{$IFDEF CPU64BITS}
  Is64: Boolean = True;
{$ELSE}
  Is64: Boolean = False;
{$ENDIF}

// ������ BPL ��ʵ�ʵķ�����ַ��֧�� 32 λ�� 64 λ�������붼Ϊ $25FF�������岻ͬ
function GetBplMethodAddress(Method: Pointer): Pointer;
type
  TJmpCode = packed record
    Code: Word;                 // �����תָ����Ϊ $25FF
{$IFDEF CPU64BITS}
    Addr: DWORD;                // 64 λ�µ���ת���� 8 �ֽڵ�ַ���洢λ�õ����ƫ�ƣ�Ҳ�� 32 λ��JMP QWORD PTR [RIP + Addr]
{$ELSE}
    Addr: ^Pointer;             // 32 λ�µ���תָ���ַ��ָ�򱣴�Ŀ���ַ��ָ�룬JMP DWORD PTR [Addr]
{$ENDIF}
  end;
  PJmpCode = ^TJmpCode;

{$IFDEF CPU64BITS}
var
  P: PPointer;
{$ENDIF}
begin
  if (Method <> nil) and (PJmpCode(Method)^.Code = csJmp32Code) then
  begin
{$IFDEF CPU64BITS}
    // Addr ���һ�� 32 λƫ�ƣ����� RIP Ҳ���� Method ����ټӱ���תָ��� 6 �ֽ�
    // ���ܵõ�һ�����Ե�ַ���õ�ַ��ŵ� 8 �ֽ�����������תĿ���ַ
    P := PPointer(NativeInt(Method) + SizeOf(TJmpCode) + Integer(PJmpCode(Method)^.Addr));
    Result := P^;
{$ELSE}
    Result := PJmpCode(Method)^.Addr^;
{$ENDIF}
  end
  else
    Result := Method;
end;

// ���� Interface ��ĳ��ŷ�����ʵ�ʵ�ַ�������� Self ƫ�ƣ�֧�� 32 λ�� 64 λ
function GetInterfaceMethodAddress(const AIntf: IUnknown;
  MethodIndex: Integer): Pointer;
type
  TIntfMethodEntry = packed record
    case Integer of
      0: (ByteOpCode: Byte);        // 32 λ�µ� $05 �����ֽ�
      1: (WordOpCode: Word);        // 32 λ�µ� $C083 ��һ�ֽ�
      2: (DWordOpCode: DWORD);      // 32 λ�µ� $04244483 ��һ�ֽڻ� $04244481 �����ֽڣ�
                                    // �� 64 λ�µ� $4883C1E0 ��һ�ֽ�
  end;
  PIntfMethodEntry = ^TIntfMethodEntry;

{$IFDEF CPU64BITS}
  TRelativeAddr = DWORD;
{$ELSE}
  TRelativeAddr = ^Pointer;
{$ENDIF}

  // ������ת�����������ʵ���ϵ�ͬ�� TJmpCode �� TLongJmp ���ṹ�����
  TIntfJumpEntry = packed record
    case Integer of
      0: (ByteOpCode: Byte; Offset: LongInt);       // $E9 �����ֽڣ�32 λ�� 64 λͨ��
      1: (WordOpCode: Word; Addr: TRelativeAddr);   // $25FF �����ֽ�
  end;
  PIntfJumpEntry = ^TIntfJumpEntry;
  PPointer = ^Pointer;

var
  OffsetStubPtr: Pointer;
  IntfPtr: PIntfMethodEntry;
  JmpPtr: PIntfJumpEntry;
{$IFDEF CPU64BITS}
  P: PPointer;
{$ENDIF}
begin
  Result := nil;
  if (AIntf = nil) or (MethodIndex < 0) then
    Exit;

  OffsetStubPtr := PPointer(TCnAddressInt(PPointer(AIntf)^) + SizeOf(Pointer) * MethodIndex)^;

  // �õ��� interface ��Ա������ת��ڣ�����ڻ����� Self ָ��������������
  // 32 λ�£�IUnknown �����׼������ھ��� add dword ptr [esp+$04],-$xx ��xx Ϊ ShortInt �� LongInt������Ϊ�� stdcall
  // stdcall/safecall/cdecl �Ĵ���Ϊ $04244483 ��һ�ֽڵ� ShortInt���� $04244481 �����ֽڵ� LongInt
  // ���������������÷�ʽ���п�����Ĭ�� register �� add eax -$xx ��xx Ϊ ShortInt �� LongInt��
  // stdcall/safecall/cdecl �Ĵ���Ϊ $C083 ��һ�ֽڵ� ShortInt���� $05 �����ֽڵ� LongInt
  // pascal ��������ջ��ʽ�����ƺ��Ժ� stdcall ��һ��
  // Win64 �£�������ھ��� add ecx, -$20��֮���� Jump
  IntfPtr := PIntfMethodEntry(OffsetStubPtr);

  JmpPtr := nil;

{$IFDEF CPU64BITS}
  // 64 λ��ת�ƺ�����һ��
  if IntfPtr^.DWordOpCode = $E0C18348 then
    JmpPtr := PIntfJumpEntry(TCnAddressInt(IntfPtr) + 4);
{$ELSE}
  if IntfPtr^.ByteOpCode = $05 then
    JmpPtr := PIntfJumpEntry(TCnAddressInt(IntfPtr) + 1 + 4)
  else if IntfPtr^.DWordOpCode = $04244481 then
    JmpPtr := PIntfJumpEntry(TCnAddressInt(IntfPtr) + 4 + 4)
  else if IntfPtr^.WordOpCode = $C083 then
    JmpPtr := PIntfJumpEntry(TCnAddressInt(IntfPtr) + 2 + 1)
  else if IntfPtr^.DWordOpCode = $04244483 then
    JmpPtr := PIntfJumpEntry(TCnAddressInt(IntfPtr) + 4 + 1);
{$ENDIF}

  if JmpPtr <> nil then
  begin
    // Ҫ���ָ��ֲ�ͬ����ת�������� E9 �����ֽ����ƫ�ƣ�32 λ�� 64 λͨ�ã����Լ� 25FF �����ֽھ��Ե�ַ�ĵ�ַ
    if JmpPtr^.ByteOpCode = csJmpCode then
    begin
      Result := Pointer(TCnAddressInt(JmpPtr) + JmpPtr^.Offset + 5); // 5 ��ʾ Jmp ָ��ĳ���
    end
    else if JmpPtr^.WordOpCode = csJmp32Code then
    begin
{$IFDEF CPU64BITS}
      // Addr ���һ�� 32 λƫ�ƣ����� RIP Ҳ���Ǳ�����ټӱ���תָ��� 6 �ֽ�
      // ���ܵõ�һ�����Ե�ַ���õ�ַ��ŵ� 8 �ֽ�����������תĿ���ַ
      P := PPointer(NativeInt(JmpPtr) + 6 + Integer(JmpPtr^.Addr));
      Result := P^;
{$ELSE}
      Result := JmpPtr^.Addr^;
{$ENDIF}
    end;
  end;
end;

//==============================================================================
// ��̬�� dynamic �����ҽ���
//==============================================================================

{ TCnMethodHook }

constructor TCnMethodHook.Create(const AOldMethod, ANewMethod: Pointer;
  UseDDteoursHook, DefaultHook: Boolean);
begin
  inherited Create;
{$IFNDEF USE_DDETOURS_HOOK}
  if UseDDteoursHook then
    raise Exception.Create(SCnErrorNoDDetours);
{$ENDIF}

  FUseDDteours := UseDDteoursHook;

  FHooked := False;
  FOldMethod := AOldMethod;
  FNewMethod := ANewMethod;
  FTrampoline := nil;

{$IFDEF CPU64BITS}
  FFar := IsUInt64SubOverflowInt32(UInt64(FNewMethod), UInt64(FOldMethod));
{$ENDIF}

  if DefaultHook then
    HookMethod;
end;

destructor TCnMethodHook.Destroy;
begin
  UnHookMethod;
  inherited;
end;

procedure TCnMethodHook.HookMethod;
var
  DummyProtection: DWORD;
  OldProtection: DWORD;
{$IFDEF CPU64BITS}
  NewAddr: UInt64;
{$ENDIF}
begin
  if FHooked then Exit;

  if FUseDDteours then
  begin
{$IFDEF USE_DDETOURS_HOOK}
    FTrampoline := DDetours.InterceptCreate(FOldMethod, FNewMethod);
    if not Assigned(FTrampoline) then
      raise Exception.Create(SCnFailInstallHook);
{$ENDIF}
  end
  else
  begin
    if Is64 {$IFDEF CPU64BITS} and FFar {$ENDIF} then
    begin
{$IFDEF CPU64BITS}
      // 64 λ����ת
      if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump64), PAGE_EXECUTE_READWRITE, @OldProtection) then
        raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);

      try
        // ����ԭ���Ĵ���
        FSaveData64 := PCnLongJump64(FOldMethod)^;

        // ����תָ���滻ԭ������ǰ 14 �ֽڴ���
        InitLongJump64(PCnLongJump64(FOldMethod));

        NewAddr := UInt64(FNewMethod); // 64 λ��ת��ַ��ɸߵ������ֱַ������ջ
        PCnLongJump64(FOldMethod)^.AddrLow32 := DWORD(NewAddr and $FFFFFFFF);
        PCnLongJump64(FOldMethod)^.AddrHigh32 := DWORD(NewAddr shr 32);

        // ����ദ������ָ�����ͬ��
        FlushInstructionCache(GetCurrentProcess, FOldMethod, SizeOf(TCnLongJump64));
      finally
        // �ָ�����ҳ����Ȩ��
        if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump64), OldProtection, @DummyProtection) then
          raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);
      end;
{$ENDIF}
    end
    else // 64 �� 32 λ�����ת
    begin
      // ���ô���ҳд����Ȩ��
      if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump), PAGE_EXECUTE_READWRITE, @OldProtection) then
        raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);

      try
        // ����ԭ���Ĵ���
        FSaveData := PCnLongJump(FOldMethod)^;

        // ����תָ���滻ԭ������ǰ 5 �ֽڴ���
        PCnLongJump(FOldMethod)^.JmpOp := csJmpCode;
{$IFDEF CPU64BITS}
        PCnLongJump(FOldMethod)^.Addr := DWORD(TCnAddressInt(FNewMethod) -
          TCnAddressInt(FOldMethod) - SizeOf(TCnLongJump)); // 64 ��Ҳʹ�� 32 λ��Ե�ַ
{$ELSE}
        PCnLongJump(FOldMethod)^.Addr := Pointer(TCnAddressInt(FNewMethod) -
          TCnAddressInt(FOldMethod) - SizeOf(TCnLongJump)); // ʹ�� 32 λ��Ե�ַ
{$ENDIF}

        // ����ദ������ָ�����ͬ��
        FlushInstructionCache(GetCurrentProcess, FOldMethod, SizeOf(TCnLongJump));
      finally
        // �ָ�����ҳ����Ȩ��
        if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump), OldProtection, @DummyProtection) then
          raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);
      end;
    end;
  end;

  FHooked := True;
end;

{$IFDEF CPU64BITS}

procedure TCnMethodHook.InitLongJump64(JmpPtr: PCnLongJump64);
begin
  if JmpPtr <> nil then
  begin
    JmpPtr^.PushOp := $68;
    JmpPtr^.MovOp := $042444C7;
    JmpPtr^.RetOp := $C3;
  end;
end;

{$ENDIF}

procedure TCnMethodHook.UnhookMethod;
var
  DummyProtection: DWORD;
  OldProtection: DWORD;
begin
  if not FHooked then Exit;

  if FUseDDteours then
  begin
{$IFDEF USE_DDETOURS_HOOK}
    if not DDetours.InterceptRemove(FTrampoline) then
      raise Exception.Create(SCnFailUninstallHook);
{$ENDIF}
    FTrampoline := nil;
  end
  else
  begin
    if Is64 {$IFDEF CPU64BITS} and FFar {$ENDIF} then
    begin
{$IFDEF CPU64BITS}
      // ���ô���ҳд����Ȩ��
      if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump64), PAGE_READWRITE, @OldProtection) then
        raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);

      try
        // �ָ�ԭ���Ĵ���
        PCnLongJump64(FOldMethod)^ := FSaveData64;
      finally
        // �ָ�����ҳ����Ȩ��
        if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump64), OldProtection, @DummyProtection) then
          raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);
      end;

      // ����ദ������ָ�����ͬ��
      FlushInstructionCache(GetCurrentProcess, FOldMethod, SizeOf(TCnLongJump64));
{$ENDIF}
    end
    else
    begin
      // ���ô���ҳд����Ȩ��
      if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump), PAGE_READWRITE, @OldProtection) then
        raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);

      try
        // �ָ�ԭ���Ĵ���
        PCnLongJump(FOldMethod)^ := FSaveData;
      finally
        // �ָ�����ҳ����Ȩ��
        if not VirtualProtect(FOldMethod, SizeOf(TCnLongJump), OldProtection, @DummyProtection) then
          raise Exception.CreateFmt(SCnMemoryWriteError, [SysErrorMessage(GetLastError)]);
      end;

      // ����ദ������ָ�����ͬ��
      FlushInstructionCache(GetCurrentProcess, FOldMethod, SizeOf(TCnLongJump));
    end;
  end;

  FHooked := False;
end;

end.

