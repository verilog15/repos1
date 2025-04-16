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

unit CnWatchFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnDebugViewer
* ��Ԫ���ƣ����Ӵ���ʵ�ֵ�Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2018.01.05
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms,
  Dialogs, Contnrs, CnMsgClasses, ComCtrls;

type
  TCnWatchItem = class(TObject)
  private
    FRecentChanged: Boolean;
    FVarName: string;
    FVarValue: string;
  public
    property VarName: string read FVarName write FVarName;
    property VarValue: string read FVarValue write FVarValue;
    property RecentChanged: Boolean read FRecentChanged write FRecentChanged;
  end;

  TCnWatchList = class(TObjectList)
  private
    function GetItem(Index: Integer): TCnWatchItem;
    procedure SetItem(Index: Integer; const Value: TCnWatchItem);
  public
    function IndexOfVarName(const VarName: string): TCnWatchItem;
    procedure DeleteVarName(const VarName: string);
    procedure ClearChanged;
    property Items[Index: Integer]: TCnWatchItem read GetItem write SetItem; default;
  end;

  TCnWatchForm = class(TForm)
    lvWatch: TListView;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure lvWatchData(Sender: TObject; Item: TListItem);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
  private
    FWatchList: TCnWatchList;
    FManualClose: Boolean;
    procedure WatchItemChanged(Sender: TObject; const VarName: string;
      const NewValue: string);
    procedure WatchItemCleared(Sender: TObject; const VarName: string);
  protected
    procedure CreateParams(var Params: TCreateParams); override;
  public
    procedure AdjustPosition;
    property ManualClose: Boolean read FManualClose write FManualClose;
    // ����û�û���ƹ��رգ����Զ���ʾ����
  end;

var
  CnWatchForm: TCnWatchForm = nil;

procedure CreateWatchForm;

implementation

{$R *.dfm}

procedure CreateWatchForm;
begin
  if CnWatchForm = nil then
  begin
    CnWatchForm := TCnWatchForm.Create(Application);
    CnWatchForm.AdjustPosition;

    SetWindowPos(CnWatchForm.Handle, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE or
      SWP_NOACTIVATE or SWP_NOSIZE or SWP_NOOWNERZORDER); // ������ʾ����ǰ��
  end;
end;

{ TCnWatchForm }

procedure TCnWatchForm.FormCreate(Sender: TObject);
begin
  FWatchList := TCnWatchList.Create(True);
  CnMsgManager.OnWatchItemCleared := WatchItemCleared;
  CnMsgManager.OnWatchItemChanged := WatchItemChanged;
end;

procedure TCnWatchForm.FormDestroy(Sender: TObject);
begin
  CnMsgManager.OnWatchItemCleared := nil;
  CnMsgManager.OnWatchItemChanged := nil;
  FWatchList.Free;
end;

procedure TCnWatchForm.WatchItemChanged(Sender: TObject; const VarName,
  NewValue: string);
var
  Item: TCnWatchItem;
begin
  if VarName <> '' then
  begin
    FWatchList.ClearChanged;
    Item := FWatchList.IndexOfVarName(VarName);
    if Item <> nil then
    begin
      Item.VarValue := NewValue;
      Item.RecentChanged := True;
    end
    else
    begin
      Item := TCnWatchItem.Create;
      Item.VarName := VarName;
      Item.VarValue := NewValue;
      Item.RecentChanged := True;
      FWatchList.Add(Item);
    end;

    lvWatch.Items.Count := FWatchList.Count;
    lvWatch.Invalidate;
  end;
end;

procedure TCnWatchForm.WatchItemCleared(Sender: TObject;
  const VarName: string);
begin
  if VarName <> '' then
  begin
    FWatchList.DeleteVarName(VarName);
    lvWatch.Items.Count := FWatchList.Count;
    lvWatch.Invalidate;
  end;
end;

procedure TCnWatchForm.AdjustPosition;
var
  WorkRect: TRect;
begin
  SystemParametersInfo(SPI_GETWORKAREA, 0, @WorkRect, 0);
  Top := WorkRect.Bottom - Height;
  Left := WorkRect.Right - Width;
end;

procedure TCnWatchForm.CreateParams(var Params: TCreateParams);
begin
  inherited;
  Params.WndParent := GetDesktopWindow;
  Params.ExStyle := Params.ExStyle or WS_EX_TOPMOST or WS_EX_TOOLWINDOW;
end;

{ TCnWatchList }

procedure TCnWatchList.ClearChanged;
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    if Items[I] <> nil then
      Items[I].RecentChanged := False;
end;

procedure TCnWatchList.DeleteVarName(const VarName: string);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
  begin
    if Items[I].VarName = VarName then
    begin
      Delete(I);
      Exit;
    end;
  end;
end;

function TCnWatchList.GetItem(Index: Integer): TCnWatchItem;
begin
  Result := TCnWatchItem(inherited Items[Index]);
end;

function TCnWatchList.IndexOfVarName(const VarName: string): TCnWatchItem;
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
  begin
    if Items[I].VarName = VarName then
    begin
      Result := Items[I];
      Exit;
    end;
  end;
  Result := nil;
end;

procedure TCnWatchList.SetItem(Index: Integer; const Value: TCnWatchItem);
begin
  inherited Items[Index] := Value;
end;

procedure TCnWatchForm.lvWatchData(Sender: TObject; Item: TListItem);
var
  Idx: Integer;
  Watch: TCnWatchItem;
begin
  Idx := Item.Index;
  if (Idx >= 0) and (Idx < FWatchList.Count) then
  begin
    Watch := FWatchList[Idx];
    if Watch <> nil then
    begin
      Item.Caption := Watch.VarName;
      Item.SubItems.Clear;
      Item.SubItems.Add(Watch.VarValue);
    end;
  end;
end;

procedure TCnWatchForm.FormClose(Sender: TObject;
  var Action: TCloseAction);
begin
  FManualClose := True;
end;

end.
