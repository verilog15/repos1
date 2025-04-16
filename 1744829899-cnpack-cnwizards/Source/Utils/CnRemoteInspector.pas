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

unit CnRemoteInspector;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ����Թ�����չ��Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin7Pro + Delphi 10.3
* ���ݲ��ԣ�����
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2023.09.05 V1.0
*               ʵ�ֵ�Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  SysUtils, Classes, Windows, CnWizDebuggerNotifier, CnPropSheetFrm;

function EvaluateRemoteExpression(const Expression: string;
  AForm: TCnPropSheetForm = nil; SyncMode: Boolean = True;
  AParentSheet: TCnPropSheetForm = nil): TCnPropSheetForm;
{* ִ�б����Ե�Զ�̽����е���ֵ�鿴}

implementation

uses
  CnNative {$IFDEF DEBUG}, CnDebug {$ENDIF};

type
  TCnRemoteEvaluationInspector = class(TCnObjectInspector)
  private
    FObjectExpr: string;
    FEvaluator: TCnRemoteProcessEvaluator;
  protected
    procedure SetObjectAddr(const Value: Pointer); override;

    procedure DoEvaluate; override;
  public
    constructor Create(Data: Pointer); override;
    destructor Destroy; override;

{$IFDEF SUPPORT_ENHANCED_RTTI}
    function ChangeFieldValue(const FieldName, Value: string;
      FieldObj: TCnFieldObject): Boolean; override;
{$ENDIF}
    function ChangePropertyValue(const PropName, Value: string;
      PropObj: TCnPropertyObject): Boolean; override;

    property ObjectExpr: string read FObjectExpr;
  end;

  // ClassInfo ָ��ýṹ
  TCnTypeInfoRec = packed record
    TypeKind: Byte;
    NameLength: Byte;
    // NameLength �� Byte �� ClassName���ٺ����� TCnTypeDataRec32/64
  end;
  PCnTypeInfoRec = ^TCnTypeInfoRec;

  TCnTypeDataRec32 = packed record
    ClassType: Cardinal;
    ParentInfo: Cardinal;
    PropCount: SmallInt;
    UnitNameLength: Byte;
    // UnitNameLength �� Byte �� UnitName���ٺ����� TCnPropDataRec
  end;
  PCnTypeDataRec32 = ^TCnTypeDataRec32;

  TCnTypeDataRec64 = packed record
    ClassType: Int64;
    ParentInfo: Int64;
    PropCount: SmallInt;
    UnitNameLength: Byte;
    // UnitNameLength �� Byte �� UnitName���ٺ����� TCnPropDataRec
  end;
  PCnTypeDataRec64 = ^TCnTypeDataRec64;

  TCnPropDataRec = packed record
    PropCount: Word;
    // �ٺ����� TCnPropInfoRec32/64 �б�
  end;
  PCnPropDataRec = ^TCnPropDataRec;

  TCnPropInfoRec32 = packed record
    PropType: Cardinal;
    GetProc: Cardinal;
    SetProc: Cardinal;
    StoredProc: Cardinal;
    Index: Integer;
    Default: Longint;
    NameIndex: SmallInt;
    NameLength: Byte;
    // NameLength �� Byte �� PropName
  end;
  PCnPropInfoRec32 = ^TCnPropInfoRec32;

  TCnPropInfoRec64 = packed record
    PropType: Int64;
    GetProc: Int64;
    SetProc: Int64;
    StoredProc: Int64;
    Index: Integer;
    Default: Longint;
    NameIndex: SmallInt;
    NameLength: Byte;
    // NameLength �� Byte �� PropName
  end;
  PCnPropInfoRec64 = ^TCnPropInfoRec64;

function EvaluateRemoteExpression(const Expression: string;
  AForm: TCnPropSheetForm; SyncMode: Boolean;
  AParentSheet: TCnPropSheetForm): TCnPropSheetForm;
var
  Eval: TCnRemoteProcessEvaluator;
begin
  Result := nil;
  if Trim(Expression) = '' then Exit;

  if AForm = nil then
    AForm := TCnPropSheetForm.Create(nil);

  AForm.ObjectPointer := nil;
  AForm.ObjectExpr := Trim(Expression); // ע���ʱ ObjectPointer ����Ϊ nil���ڲ��ж�ʹ��
  AForm.Clear;
  AForm.ParentSheetForm := AParentSheet;

  AForm.SyncMode := SyncMode;
  AForm.InspectorClass := TCnRemoteEvaluationInspector;

  Eval := TCnRemoteProcessEvaluator.Create;
  if SyncMode then
  begin
    AForm.DoEvaluateBegin;
    try
      AForm.InspectParam := Eval;
      AForm.InspectObject(AForm.InspectParam);
    finally
      AForm.DoEvaluateEnd;
      AForm.Show;  // After Evaluation. Show the form.
    end;
  end
  else
    PostMessage(AForm.Handle, CN_INSPECTOBJECT, WParam(Eval), 0);

  Result := AForm;
end;

{ TCnRemoteEvaluationInspector }

{$IFDEF SUPPORT_ENHANCED_RTTI}

function TCnRemoteEvaluationInspector.ChangeFieldValue(const FieldName,
  Value: string; FieldObj: TCnFieldObject): Boolean;
begin

end;

{$ENDIF}

function TCnRemoteEvaluationInspector.ChangePropertyValue(const PropName,
  Value: string; PropObj: TCnPropertyObject): Boolean;
begin

end;

constructor TCnRemoteEvaluationInspector.Create(Data: Pointer);
begin
  inherited Create(Data);
  FEvaluator := TCnRemoteProcessEvaluator(Data);
end;

destructor TCnRemoteEvaluationInspector.Destroy;
begin
  FEvaluator.Free;
  inherited;
end;

{
  ������ʽ�Ƕ��������� ClassInfo �õ���ַָ�룬��һ�θ��� 256 + 256 �ֽڣ������õ����������� Info ָ������������
  �ٸ������������������� ClassInfo �� 256 * �������� + 1���ֽڣ��õ�������������������ȡ
  �ٶ������丸�� Info ָ�룬�ٶ� 256 * �������� + 1���ֽڣ��õ���������������������������ȡ
  �ܹ������㹻���������� TObject ��ͣ�����Ҫ���ö���Ĳ㼶����ô��εĵ�ַ�ռ�����
}
procedure TCnRemoteEvaluationInspector.DoEvaluate;
var
  I, Len, Ret, APCnt, PCnt: Integer;
  BufLen: Integer;
  RemPtr: TCnOTAAddress;
  Base, S: string;
  Buf: TBytes;
  BufPtr: PByte;
  Hies: TStringList;
  AProp: TCnPropertyObject;
  Is32: Boolean;
begin
  if FObjectExpr = '' then
  begin
    InspectComplete := True;
    Exit;
  end;

  if not IsRefresh then
  begin
    Properties.Clear;
    Fields.Clear;
    Events.Clear;
    Methods.Clear;
    Components.Clear;
    Controls.Clear;
    CollectionItems.Clear;
    Strings.Clear;
    MenuItems.Clear;
    Graphics.Graphic := nil;
  end;

  if not CnWizDebuggerObjectInheritsFrom(FObjectExpr, 'TObject', FEvaluator) then
  begin
    InspectComplete := True;
    Exit;
  end;

  if CnWizDebuggerObjectInheritsFrom(FObjectExpr, 'TStrings', FEvaluator) then
  begin
    ContentTypes := ContentTypes + [pctStrings];
    S := FEvaluator.EvaluateExpression('(' + FObjectExpr + ' as TStrings).Text');
    if Strings.DisplayValue <> S then
    begin
      Strings.Changed := True;
      Strings.DisplayValue := S;
    end;
  end;

  Base := Format('(%s)', [FObjectExpr]);
  S := Format('Pointer(%s.ClassInfo)', [Base]);
  S := FEvaluator.EvaluateExpression(S);
  if LowerCase(S) = 'nil' then
    RemPtr := 0
  else
    RemPtr := StrToUInt64(S);

  if RemPtr <> 0 then
  begin
    ContentTypes := [pctHierarchy];
    Hies := TStringList.Create;
    try
      BufLen := 512;
      SetLength(Buf, BufLen);
      Ret := FEvaluator.ReadProcessMemory(RemPtr, BufLen, Buf[0]);
{$IFDEF DEBUG}
      CnDebugger.LogFmt('FEvaluator.ReadProcessMemory %d Return %d', [BufLen, Ret]);
      CnDebugger.LogMemDump(@Buf[0], Ret);
{$ENDIF}

      BufPtr := @Buf[0];
      Len := PCnTypeInfoRec(BufPtr)^.NameLength;
      Inc(BufPtr, SizeOf(TCnTypeInfoRec));                      // �������ֽ�ָ�� ClassName
      Inc(BufPtr, Len);                                         // �������ַ�������ָ�� TypeData

      // ��������������Buf �� TCnTypeInfoRec
      Is32 := FEvaluator.CurrentProcessIs32;
{$IFDEF DEBUG}
      if Is32 then
        CnDebugger.LogMsg('Remote Process is 32.')
      else
        CnDebugger.LogMsg('Remote Process is 64.');
{$ENDIF}

      if Is32 then
        APCnt := PCnTypeDataRec32(BufPtr)^.PropCount
      else
        APCnt := PCnTypeDataRec64(BufPtr)^.PropCount;

{$IFDEF DEBUG}
      CnDebugger.LogFmt('All Properties Count: %d', [APCnt]);
{$ENDIF}

      if APCnt > 0 then
      begin
        // �õ�����������BufLen ����ȷ�������¶��� Buf����׼���� BufPtr����ʼѭ��
        BufLen := (APCnt + 1) * 256; // Ԥ�������ܴ�Ŀռ�
        SetLength(Buf, BufLen);
        Ret := FEvaluator.ReadProcessMemory(RemPtr, BufLen, Buf[0]);
{$IFDEF DEBUG}
        CnDebugger.LogFmt('FEvaluator.ReadProcessMemory %d Return %d', [BufLen, Ret]);
{$ENDIF}
        // Buf �Ǳ���� PCnTypeInfoRec
        BufPtr := @Buf[0];

        repeat
          Len := PCnTypeInfoRec(BufPtr)^.NameLength;
          Inc(BufPtr, SizeOf(TCnTypeInfoRec));                      // �������ֽ�ָ�� ClassName

          SetLength(S, Len);
          Move(BufPtr^, S[1], Len);
{$IFDEF DEBUG}
          CnDebugger.LogFmt('ClassName: %s', [S]);
{$ENDIF}
          Hies.Add(S);

          Inc(BufPtr, Len);                                         // �������ַ�������ָ�� TypeData

          Is32 := FEvaluator.CurrentProcessIs32;
          RemPtr := 0;

          Base := Base + '.ClassParent';                            // ���ø���� ClassInfo ����
          S := Format('Pointer(%s.ClassInfo)', [Base]);
          S := FEvaluator.EvaluateExpression(S);
          if LowerCase(S) <> 'nil' then
            RemPtr := StrToUInt64(S);

          if Is32 then
          begin
            // APCnt := PCnTypeDataRec32(BufPtr)^.PropCount;           // �õ����ൽ�����������������
            Len := PCnTypeDataRec32(BufPtr)^.UnitNameLength;
            Inc(BufPtr, SizeOf(TCnTypeDataRec32));                  // ָ�� UnitName �ַ���
            SetLength(S, Len);
            Move(BufPtr^, S[1], Len);
{$IFDEF DEBUG}
            CnDebugger.LogFmt('UnitName: %s', [S]);
{$ENDIF}
          end
          else
          begin
            // APCnt := PCnTypeDataRec64(BufPtr)^.PropCount;
            Len := PCnTypeDataRec64(BufPtr)^.UnitNameLength;
            Inc(BufPtr, SizeOf(TCnTypeDataRec64));                  // ָ�� UnitName �ַ���
            SetLength(S, Len);
            Move(BufPtr^, S[1], Len);
{$IFDEF DEBUG}
            CnDebugger.LogFmt('UnitName: %s', [S]);
{$ENDIF}
          end;

          Inc(BufPtr, Len);                                         // ���� UnitName ָ�� PropData
          PCnt := PCnPropDataRec(BufPtr)^.PropCount;                // �õ������������
{$IFDEF DEBUG}
          CnDebugger.LogFmt('Properties Count: %d', [PCnt]);
{$ENDIF}
          Inc(BufPtr, SizeOf(TCnPropDataRec));                      // ָ�� PropInfo������еĻ�

          if PCnt > 0 then
          begin
            for I := 0 to PCnt - 1 do
            begin
              if Is32 then
              begin
                Len := PCnPropInfoRec32(BufPtr)^.NameLength;        // �õ��������ĳ���
                Inc(BufPtr, SizeOf(TCnPropInfoRec32));              // BufPtr ָ��������
                SetLength(S, Len);
                Move(BufPtr^, S[1], Len);                           // ����������
                Inc(BufPtr, Len);                                   // BufPtr �������ƣ�ָ����һ��
              end
              else
              begin
                Len := PCnPropInfoRec64(BufPtr)^.NameLength;        // �õ��������ĳ���
                Inc(BufPtr, SizeOf(TCnPropInfoRec64));              // BufPtr ָ��������
                SetLength(S, Len);
                Move(BufPtr^, S[1], Len);                           // ����������
                Inc(BufPtr, Len);                                   // BufPtr �������ƣ�ָ����һ��
              end;
              // �õ��������� S ��
{$IFDEF DEBUG}
              CnDebugger.LogFmt('Property %d: %s', [I + 1, S]);
{$ENDIF}
            end;
          end;

          // ����õ��˸�����Ϣ��Զ��ָ�룬�����¶���һ���ڴ�
          if RemPtr <> 0 then
          begin
            Ret := FEvaluator.ReadProcessMemory(RemPtr, BufLen, Buf[0]);
{$IFDEF DEBUG}
            CnDebugger.LogFmt('FEvaluator.ReadProcessMemory %d Return %d', [BufLen, Ret]);
{$ENDIF}
            BufPtr := @Buf[0];                                        // ָ���࣬���¿�ʼѭ��
          end
          else
            BufPtr := nil;
        until BufPtr = nil;
      end;

      Hierarchy := Hies.Text;
      DoAfterEvaluateHierarchy;
    finally
      Hies.Free;
    end;
  end;

//{$IFDEF SUPPORT_ENHANCED_RTTI}
//  S := Format('Length(TRttiContext.Create.GetType(%s.ClassInfo).GetProperties)', [FObjectExpr]);
//  S := FEvaluator.EvaluateExpression(S);
//  C := StrToIntDef(S, 0);
//  if C > 0 then
//  begin
//    for I := 0 to C - 1 do
//    begin
//      S := FEvaluator.EvaluateExpression(Format('TRttiContext.Create.GetType(%s.ClassInfo).GetProperties[%d].PropertyType.TypeKind', [FObjectExpr, I]));
//      // �õ���������
//      if (S <> 'tkMethod') and (S <> 'tkUnknown') then
//      begin
//        // ������
//        V := FEvaluator.EvaluateExpression(Format('TRttiContext.Create.GetType(%s.ClassInfo).GetProperties[%d].Name', [FObjectExpr, I]));
//
//        // V �õ�����
//        if not IsRefresh then
//        begin
//          AProp := TCnPropertyObject.Create;
//          AProp.IsNewRTTI := True;
//        end
//        else
//          AProp := IndexOfProperty(Properties, V);
//
//        AProp.PropName := V;
//        // AProp.PropType := S;
//
//        S := FEvaluator.EvaluateExpression(Format('TRttiContext.Create.GetType(%s.ClassInfo).GetProperties[%d].GetValue(%s)', [FObjectExpr, I, FObjectExpr]));
//        if S <> AProp.DisplayValue then
//        begin
//          AProp.DisplayValue := S;
//          AProp.Changed := True;
//        end
//        else
//          AProp.Changed := False;
//
//        if not IsRefresh then
//          Properties.Add(AProp);
//
//        ContentTypes := ContentTypes + [pctProps];
//      end;
//    end;
//  end;
//{$ENDIF}

  InspectComplete := True;
end;

procedure TCnRemoteEvaluationInspector.SetObjectAddr(const Value: Pointer);
var
  L: Integer;
begin
  inherited;
  if Value = nil then
    FObjectExpr := ''
  else
  begin
    L := StrLen(PChar(Value));
    if L > 0 then
    begin
      SetLength(FObjectExpr, L);
      Move(Value^, FObjectExpr[1], L * SizeOf(Char));
    end;
  end;
end;

end.
