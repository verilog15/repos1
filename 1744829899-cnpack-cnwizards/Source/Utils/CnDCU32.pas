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

unit CnDCU32;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�DCU32 �򵥷�װ��Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��
* ����ƽ̨��PWinXP SP2 + Delphi 5
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6
* �� �� �����õ�Ԫ���ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2005.08.11 v1.0
*             ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNUSESTOOLS}

uses
  Windows, Classes, SysUtils, Contnrs, DCURecs, DCU32, DCU_Out
  {$IFNDEF STAND_ALONE}, ToolsAPI, CnWizUtils, CnPasCodeParser
  {$IFDEF UNICODE}, CnWidePasParser {$ENDIF}, CnCommon,
  CnWizConsts {$ENDIF};

type

{ TCnUnitUsesInfo }

  TCnUnitUsesInfo = class(TUnit)
  {* ������ DCU �н��������� Uses ����}
  private
    FIntfUses: TStringList;  // �洢 interface ���ֵ� uses ��Ԫ���Լ�ÿ����Ԫ��Ӧ�ĵ�������
    FImplUses: TStringList;  // �洢 implementation ���ֵ� uses ��Ԫ���Լ�ÿ����Ԫ��Ӧ�ĵ�������
    function GetImplUse(Index: Integer): string;
    function GetImplUsesCount: Integer;
    function GetImplUsesImport(Index: Integer): TStrings;
    function GetIntfUse(Index: Integer): string;
    function GetIntfUsesCount: Integer;
    function GetIntfUsesImport(Index: Integer): TStrings;
    procedure GetUsesList(AList: TStringList; AFlag: TUnitImpFlags);
    procedure ClearUsesList(AList: TStringList);
  public
    constructor Create(const DcuName: string; UseOnly: Boolean = True); reintroduce;
    destructor Destroy; override;
    procedure Sort;
    
    property IntfUsesCount: Integer read GetIntfUsesCount;
    {* interface �����ж��ٸ� uses}
    property IntfUses[Index: Integer]: string read GetIntfUse;
    {* interface ���ֵ�ÿһ�� uses}
    property IntfUsesImport[Index: Integer]: TStrings read GetIntfUsesImport;
    {* interface ����ÿһ�� uses ��Ԫ����������б�}
    property ImplUsesCount: Integer read GetImplUsesCount;
    {* implementation �����ж��ٸ� uses}
    property ImplUses[Index: Integer]: string read GetImplUse;
    {* implementation ���ֵ�ÿһ�� uses}
    property ImplUsesImport[Index: Integer]: TStrings read GetImplUsesImport;
    {* implementation ����ÿһ�� uses ��Ԫ����������б�}

    property ExportedNames: TStringList read FExportNames;
    {* ������������}
  end;

  TCnUsesKind = (ukHasInitSection, ukHasRegProc, ukInCleanList, ukInIgnoreList,
    ukNoSource, tkCompRef);
  TCnUsesKinds = set of TCnUsesKind;

{ TCnUsesItem }

  TCnUsesItem = class
  {* ����һ�� uses ��}
  private
    FChecked: Boolean;
    FKinds: TCnUsesKinds;
    FName: string;
  public
    property Name: string read FName write FName;
    property Checked: Boolean read FChecked write FChecked;
    property Kinds: TCnUsesKinds read FKinds write FKinds;
  end;

{ TCnEmptyUsesInfo }

  TCnEmptyUsesInfo = class
  {* ����һ���账����ļ�������}
  private
    FSourceFileName: string;
{$IFNDEF STAND_ALONE}
    FProject: IOTAProject;
{$ENDIF}
    FDcuName: string;
    FIntfItems: TObjectList;
    FImplItems: TObjectList;

    function GetImplCount: Integer;
    function GetImplItem(Index: Integer): TCnUsesItem;
    function GetIntfCount: Integer;
    function GetIntfItem(Index: Integer): TCnUsesItem;
  public
    constructor Create(const ADcuName, ASourceFileName: string {$IFNDEF STAND_ALONE};
      AProject: IOTAProject {$ENDIF});
    destructor Destroy; override;
{$IFNDEF STAND_ALONE}
    function Process: Boolean;
{$ENDIF}
    property DcuName: string read FDcuName;
    property SourceFileName: string read FSourceFileName;
{$IFNDEF STAND_ALONE}
    property Project: IOTAProject read FProject;
{$ENDIF}
    property IntfCount: Integer read GetIntfCount;
    property IntfItems[Index: Integer]: TCnUsesItem read GetIntfItem;
    property ImplCount: Integer read GetImplCount;
    property ImplItems[Index: Integer]: TCnUsesItem read GetImplItem;
  end;

{$ENDIF CNWIZARDS_CNUSESTOOLS}

implementation

{$IFDEF CNWIZARDS_CNUSESTOOLS}

{$IFNDEF STAND_ALONE}
uses
  CnWizEditFiler;
{$ENDIF}

{ TCnUnitUsesInfo }

procedure TCnUnitUsesInfo.ClearUsesList(AList: TStringList);
var
  i: Integer;
begin
  if Assigned(AList) then
    for i := AList.Count - 1 downto 0 do
    begin
      AList.Objects[i].Free;
      AList.Delete(i);
    end;
end;

constructor TCnUnitUsesInfo.Create(const DcuName: string; UseOnly: Boolean);
begin
  FIntfUses := TStringList.Create;
  FImplUses := TStringList.Create;
  inherited Create;
  try
    Load(DcuName, 0, False, dcuplWin32, nil);
  except
    {$IFNDEF DELPHI2009_UP}
    raise;
    {$ENDIF}
  end;
  GetUsesList(FIntfUses, []);
  GetUsesList(FImplUses, [ufImpl]);
end;

destructor TCnUnitUsesInfo.Destroy;
begin
  ClearUsesList(FIntfUses);
  ClearUsesList(FImplUses);
  FIntfUses.Free;
  FImplUses.Free;
  inherited;
end;

function TCnUnitUsesInfo.GetImplUse(Index: Integer): string;
begin
  Result := FImplUses[Index];
end;

function TCnUnitUsesInfo.GetImplUsesCount: Integer;
begin
  Result := FImplUses.Count;
end;

function TCnUnitUsesInfo.GetImplUsesImport(Index: Integer): TStrings;
begin
  Result := TStrings(FImplUses.Objects[Index]);
end;

function TCnUnitUsesInfo.GetIntfUse(Index: Integer): string;
begin
  Result := FIntfUses[Index];
end;

function TCnUnitUsesInfo.GetIntfUsesCount: Integer;
begin
  Result := FIntfUses.Count;
end;

function TCnUnitUsesInfo.GetIntfUsesImport(Index: Integer): TStrings;
begin
  Result := TStrings(FIntfUses.Objects[Index]);
end;

procedure TCnUnitUsesInfo.GetUsesList(AList: TStringList; AFlag: TUnitImpFlags);
var
  i: Integer;
  PRec: PUnitImpRec;
  Lines: TStringList;
  Decl: TBaseDef;
begin
  ClearUsesList(AList);
  if FUnitImp.Count = 0 then
    Exit;

  for i := 0 to FUnitImp.Count - 1 do
  begin
    PRec := FUnitImp[i];
    if AFlag <> PRec.Flags then
      Continue;
    Lines := TStringList.Create;
    AList.AddObject({$IFDEF UNICODE}string{$ENDIF}(PRec^.Name^.GetStr), Lines);

    Decl := PRec^.Decls;
    while Decl <> nil do
    begin
      if Decl is TImpDef then
        Lines.Add(TImpDef(Decl).ik + ':' + {$IFDEF UNICODE}string{$ENDIF}(Decl.Name^.GetStr))
      else
        Lines.Add({$IFDEF UNICODE}string{$ENDIF}(Decl.Name^.GetStr));
      Decl := Decl.Next as TBaseDef;
    end;
  end;
end;

procedure TCnUnitUsesInfo.Sort;
begin
  FIntfUses.Sorted := True;
  FImplUses.Sorted := True;
end;

{ TCnEmptyUsesInfo }

constructor TCnEmptyUsesInfo.Create(const ADcuName, ASourceFileName: string
  {$IFNDEF STAND_ALONE}; AProject: IOTAProject {$ENDIF});
begin
  inherited Create;
  FIntfItems := TObjectList.Create;
  FImplItems := TObjectList.Create;
  FDcuName := ADcuName;
  FSourceFileName := ASourceFileName;
{$IFNDEF STAND_ALONE}
  FProject := AProject;
{$ENDIF}
end;

destructor TCnEmptyUsesInfo.Destroy;
begin
  FIntfItems.Free;
  FImplItems.Free;
  inherited;
end;

{$IFNDEF STAND_ALONE}

function TCnEmptyUsesInfo.Process: Boolean;
var
  Info: TCnUnitUsesInfo; // �洢 DCU �ļ��н��������� uses �����б�
  UsesList: TStringList; // �洢Դ���н��������� uses �����б�
  Stream: TMemoryStream;
  Item: TCnUsesItem;
  I: Integer;

  function UnitUsesListContainsUnitName({$IFNDEF SUPPORT_UNITNAME_DOT} const
    {$ENDIF} DcuName: string): Boolean;
{$IFDEF SUPPORT_UNITNAME_DOT}
  var
    K: Integer;
{$ENDIF}
  begin
    Result := UsesList.IndexOf(DcuName) >= 0;
{$IFDEF SUPPORT_UNITNAME_DOT}
    // ���û�ҵ������жϵ�ţ�ʹ�� DcuName ���һ����ĺ��沿������
    if not Result then
    begin
      K := LastCharPos(DcuName, '.');
      if K > 0 then
      begin
        Delete(DcuName, 1, K);
        Result := UsesList.IndexOf(DcuName) >= 0;
      end;
    end;
{$ENDIF}
  end;

begin
  Result := False;
  try
    Info := TCnUnitUsesInfo.Create(DcuName);
    try
      Info.Sort;
      UsesList := TStringList.Create;
      try
        Stream := TMemoryStream.Create;
        try
          EditFilerSaveFileToStream(FSourceFileName, Stream, True); // Ansi/Ansi/Utf16
{$IFDEF UNICODE}
          ParseUnitUsesW(PChar(Stream.Memory), UsesList);
{$ELSE}
          ParseUnitUses(PAnsiChar(Stream.Memory), UsesList);
{$ENDIF}
        finally
          Stream.Free;
        end;

        // ע���Դ���� UsesList ���������ĵ�Ԫ�������ǲ�����ģ�
        // �� Dcu ����������� Info ������Ǵ���ģ���ƥ�䲻��
        for I := 0 to Info.IntfUsesCount - 1 do
        begin
          if (Info.IntfUsesImport[I].Count = 0) and
            UnitUsesListContainsUnitName(Info.IntfUses[I]) then
          begin
            Item := TCnUsesItem.Create;
            Item.Name := Info.IntfUses[I];
            FIntfItems.Add(Item);
          end;
        end;

        for I := 0 to Info.ImplUsesCount - 1 do
        begin
          if (Info.ImplUsesImport[I].Count = 0) and
            UnitUsesListContainsUnitName(Info.ImplUses[I]) then
          begin
            Item := TCnUsesItem.Create;
            Item.Name := Info.ImplUses[I];
            FImplItems.Add(Item);
          end;
        end;
        Result := True;
      finally
        UsesList.Free;
      end;
    finally
      Info.Free;
    end;
  except
    on E: Exception do
      DoHandleException('Dcu32 UsesInfo ' + E.Message);
  end;
end;

{$ENDIF}

function TCnEmptyUsesInfo.GetImplCount: Integer;
begin
  Result := FImplItems.Count;
end;

function TCnEmptyUsesInfo.GetImplItem(Index: Integer): TCnUsesItem;
begin
  Result := TCnUsesItem(FImplItems[Index]);
end;

function TCnEmptyUsesInfo.GetIntfCount: Integer;
begin
  Result := FIntfItems.Count;
end;

function TCnEmptyUsesInfo.GetIntfItem(Index: Integer): TCnUsesItem;
begin
  Result := TCnUsesItem(FIntfItems[Index]);
end;

{$ENDIF CNWIZARDS_CNUSESTOOLS}
end.

