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

unit CnWizIdeHooks;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�ImageList ���� Hook ��Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע��VCL �� TCustomImageList �ṩ�� BeginUpdate �� EndUpdate �� private
*           �������� IDE �����Ҫ�������ͼƬ��ÿ����Ӷ�������ܶ�ؼ���ˢ�£�
*           ����Ӱ���ٶȡ��ر����� Delphi7 �£����һ��ͼƬ��Ҫ 70ms����ר������
*           ����Ӱ��ܴ󣬹ʱ�д�� Hook���ṩ BeginUpdate �� EndUpdate ���ܡ�
*           ���⣬ActionList Ҳ�����Ƶ����⣬ͬ������
*           ���������������˵�ʱ���е� ImageList �� ActionList ʵ����������ֻ
*           ���� IDE ���� ImageList �� ActionList
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2004.12.25 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Classes, SysUtils, Controls, ImgList, ActnList, CnWizMethodHook,
  CnWizUtils, CnWizIdeUtils;

// ��ʼ���� ImageList �� ActionList
procedure CnListBeginUpdate;

// ��������
procedure CnListEndUpdate;

implementation

{$IFDEF DEBUG}
uses
  CnDebug;
{$ENDIF}

type
{$IFNDEF IMAGELIST_BEGINENDUPDATE}
  TImageListAccess = class(TCustomImageList);
{$ENDIF}
  TActionListAccess = class(TCustomActionList);
  TListChangeProc = procedure(Self: TCustomImageList);
  TListChangeMethod = procedure of object;

  TCnListComponent = class(TComponent)
  protected
    procedure Notification(AComponent: TComponent;
      Operation: TOperation); override;
  end;
  
var
{$IFNDEF IMAGELIST_BEGINENDUPDATE}
  FImageLists: TThreadList = nil;
  FImageListHook: TCnMethodHook = nil;
{$ENDIF}
  FActionLists: TThreadList = nil;
  FActionListHook: TCnMethodHook = nil;
  FCnListComponent: TCnListComponent = nil;
  FUpdateCount: Integer = 0;

procedure TCnListComponent.Notification(AComponent: TComponent;
  Operation: TOperation);
begin
  inherited;
{$IFNDEF IMAGELIST_BEGINENDUPDATE}
  FImageLists.Remove(AComponent);
{$ENDIF}
  FActionLists.Remove(AComponent);
end;

{$IFNDEF IMAGELIST_BEGINENDUPDATE}

procedure MyImageListChange(Self: TCustomImageList);
begin
  if (Self <> nil) and (Self is TCustomImageList) then
  begin
    Self.FreeNotification(FCnListComponent);
    FImageLists.Add(Self);

// ��֪ͨ������ȥ����IDE ���� ImageList ��������£��ò���֪ͨ��
//    if Self = GetIDEImageList then
//      ClearIDEBigImageList;
  end;
end;

{$ENDIF}

procedure MyActionListChange(Self: TCustomActionList);
begin
  if (Self <> nil) and (Self is TCustomActionList) then
  begin
    Self.FreeNotification(FCnListComponent);
    FActionLists.Add(Self);
  end;
end;

procedure CnListBeginUpdate;
var
  Method: TListChangeMethod;
begin
  if FUpdateCount = 0 then
  begin
{$IFNDEF IMAGELIST_BEGINENDUPDATE}
    FImageLists := TThreadList.Create;
    FImageLists.Duplicates := dupIgnore;

    Method := TImageListAccess(GetIDEImageList).Change;
    FImageListHook := TCnMethodHook.Create(GetBplMethodAddress(TMethod(Method).Code),
      @MyImageListChange);
{$ELSE}
    if GetIDEImageList <> nil then
      GetIDEImageList.BeginUpdate;
{$ENDIF}

    FActionLists := TThreadList.Create;
    FActionLists.Duplicates := dupIgnore;

    FCnListComponent := TCnListComponent.Create(nil);
      
    Method := TActionListAccess(GetIDEActionList).Change;
    FActionListHook := TCnMethodHook.Create(GetBplMethodAddress(TMethod(Method).Code),
      @MyActionListChange);
  end;
  
  Inc(FUpdateCount);
end;

procedure CnListEndUpdate;
var
  I: Integer;
begin
  Dec(FUpdateCount);

  if FUpdateCount = 0 then
  begin
{$IFNDEF IMAGELIST_BEGINENDUPDATE}
    FreeAndNil(FImageListHook);
{$ENDIF}
    FreeAndNil(FActionListHook);
    FreeAndNil(FCnListComponent);

{$IFNDEF IMAGELIST_BEGINENDUPDATE}
    with FImageLists.LockList do
    try
      for I := Count - 1 downto 0 do
        TImageListAccess(Items[I]).Change;
    finally
      FImageLists.UnlockList;
    end;
{$ELSE}
    if GetIDEImageList <> nil then
      GetIDEImageList.EndUpdate;
{$ENDIF}

    with FActionLists.LockList do
    try
      for I := Count - 1 downto 0 do
        TActionListAccess(Items[I]).Change;
    finally
      FActionLists.UnlockList;
    end;

{$IFNDEF IMAGELIST_BEGINENDUPDATE}
    FreeAndNil(FImageLists);
{$ENDIF}
    FreeAndNil(FActionLists);
  end;
end;

end.
