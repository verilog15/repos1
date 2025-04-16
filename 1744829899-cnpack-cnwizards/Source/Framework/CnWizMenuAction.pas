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

unit CnWizMenuAction;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�IDE Action ��װ��͹�����ʵ�ֵ�Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ��    ע���õ�ԪΪ CnWizards ��ܵ�һ���֣�ʵ������չ�� IDE Action ��� Action
*           �б����Ĺ��ܡ��ⲿ�ֹ�����Ҫ�� TCnActionWizard ר�Ҽ�����ʹ�á�
*             - �����Ҫ�� IDE �����д���һ�� Action��ʹ�� WizActionMgr.Add(...)
*               ������һ�� IDE Action ����ע�������汾�� Add ���ز�ͬ�Ķ���
*             - ��������Ҫ Action ʱ������ WizActionMgr.Delete(...) ��ɾ��������
*               ��Ҫ�Լ�ȥ�ͷ� Action ����
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2012.09.19 by shenloqi
*               ��ֲ�� Delphi XE3
*           2002.09.17 V1.0
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Messages, Classes, SysUtils, Graphics, Menus, Forms, ActnList, ToolsAPI, 
  {$IFDEF DELPHIXE3_UP} Actions,{$ENDIF}
  {$IFDEF IDE_SUPPORT_HDPI} Vcl.VirtualImageList, {$ENDIF}
  CnCommon, CnWizConsts, CnWizShortCut;

type
//==============================================================================
// CnWizards IDE Action ��װ��
//==============================================================================

{ TCnWizAction }

  TCnWizAction = class(TAction)
  {* ���� CnWizards ר���е� Action �࣬������ IDE ��ݼ���װ��ͼ��ȹ��ܡ�
     �벻Ҫֱ�Ӵ������ͷŸ����ʵ������Ӧ��ʹ�� Action ������ WizActionMgr
     �� Add �� Delete ������ʵ�֡���Ҫע����ǣ�TCnWizAction ���¶�����
     ShortCut ���ԣ�����������ʵ��ת��Ϊ TAction ��д ShortCut �ǲ���ɹ��ġ�}
  private
    FCommand: string;
    FWizShortCut: TCnWizShortCut;
    FIcon: TIcon;
    FUpdating: Boolean;
    FLastUpdateTick: Cardinal;
    procedure SetInheritedShortCut;
    function GetShortCut: TShortCut;
    procedure {$IFDEF DelphiXE3_UP}_CnSetShortCut{$ELSE}SetShortCut{$ENDIF}(const Value: TShortCut);
    {* Delphi XE3 ������ SetShortCut ��������Ϊ����ͬ����������⣬�ʽ��˷�������}
    procedure OnShortCut(Sender: TObject);
  protected
    procedure Change; override;
    property WizShortCut: TCnWizShortCut read FWizShortCut;
  public
    constructor Create(AOwner: TComponent); override;
    {* �๹�������벻Ҫֱ�ӵ��ø÷���������ʵ������Ӧ���� Action ������
       WizActionMgr.Add ���������������� Delete ��ɾ����}
    destructor Destroy; override;
    {* �����������벻Ҫֱ���ͷŸ����ʵ������Ӧ���� Action ������
       WizShortCutMgr.Delete ��ɾ��һ�� Action ����}
    function Update: Boolean; override;
    {* ���� Action ״̬��}
    property Command: string read FCommand;
    {* Action �����ַ���������Ψһ��ʶһ�� Action��ͬʱҲ�ǿ�ݼ����������}
    property Icon: TIcon read FIcon;
    {* Action ������ͼ�꣬����ʱ 16x16�����������ط�ʹ�ã����벻Ҫ����ͼ������}
    property ShortCut: TShortCut read GetShortCut write {$IFDEF DelphiXE3_UP}_CnSetShortCut{$ELSE}SetShortCut{$ENDIF};
    {* Action �����Ŀ�ݼ�}
  end;

//==============================================================================
// ���˵���� CnWizards IDE Action ��װ��
//==============================================================================

{ TCnWizMenuAction }

  TCnWizMenuAction = class(TCnWizAction)
  {* ���˵������� CnMenuWizards ר���е� Action �࣬�� TCnWizAction �Ļ�������
     ���˲˵������ԡ�}
  private
    FMenu: TMenuItem;
  public
    constructor Create(AOwner: TComponent); override;
    {* �๹�������벻Ҫֱ�ӵ��ø÷���������ʵ������Ӧ���� Action ������
       WizActionMgr.Add ���������������� Delete ��ɾ����}
    destructor Destroy; override;
    {* �����������벻Ҫֱ���ͷŸ����ʵ������Ӧ���� Action ������
       WizShortCutMgr.Delete ��ɾ��һ�� Action ����}
    property Menu: TMenuItem read FMenu;
    {* Action �����Ĳ˵�����������ط�ʹ��}
  end;

//==============================================================================
// CnWizards IDE Action ��������
//==============================================================================

{ TCnWizActionMgr }

  TCnWizActionMgr = class(TComponent)
  {* IDE Action �������࣬����ά�������� IDE �е� Action �б�
     �벻Ҫֱ�Ӵ��������ʵ����Ӧʹ�� WizActionMgr ����õ�ǰ�Ĺ�����ʵ����}
  private
    FWizActions: TList;
    FWizMenuActions: TList;
    FMoreAction: TAction;
    FDeleting: Boolean;
    function GetIdeActions(Index: Integer): TContainedAction;
    function GetWizActions(Index: Integer): TCnWizAction;
    function GetWizMenuActions(Index: Integer): TCnWizMenuAction;
    function GetIdeActionCount: Integer;
    function GetWizActionCount: Integer;
    function GetWizMenuActionCount: Integer;
    procedure MoreActionExecute(Sender: TObject);
  protected
    procedure InitAction(AWizAction: TCnWizAction; const ACommand,
      ACaption: string; OnExecute: TNotifyEvent; OnUpdate: TNotifyEvent;
      const IcoName, AHint: string; UseDefaultIcon: Boolean = False);
    procedure Notification(AComponent: TComponent;
      Operation: TOperation); override;
  public
    constructor Create(AOwner: TComponent); override;
    {* �๹�������벻Ҫֱ�Ӵ��������ʵ����Ӧʹ�� WizActionMgr ����õ�ǰ
       �Ĺ�����ʵ����}
    destructor Destroy; override;
    {* ����������}
    function AddAction(const ACommand, ACaption: string; AShortCut: TShortCut;
      OnExecute: TNotifyEvent; const IcoName: string;
      const AHint: string = ''; UseDefaultIcon: Boolean = False): TCnWizAction;
    {* ����������һ�� CnWizards Action ����ͬʱ�������ӵ��б��С�
       ʹ�� Add �����Ķ���Ӧ���� Delete �������ͷš�
     |<PRE>
       ACommand: string         - Action �����֣�����Ϊһ��Ψһ���ַ���ֵ
       ACaption: string         - Action �ı���
       AShortCut: TShortCut     - Action ��Ĭ�Ͽ�ݼ���ʵ��ʹ�õļ�ֵ���ע����ж�ȡ
       OnExecute: TNotifyEvent  - ִ��֪ͨ�¼�
       IcoName: string          - Action ������ͼ������֣�����ʱ���Զ�����Դ���ļ��в���װ��
       AHint: string            - Action ����ʾ��Ϣ
       UseDefaultIcon: Boolean  - Action �Ҳ���ͼ��ʱ�Ƿ�ʹ��Ĭ��ͼ��
       Result: TCnWizAction     - ���ؽ��Ϊһ�� TCnWizAction ʵ��
     |</PRE>}
    function AddMenuAction(const ACommand, ACaption, AMenuName: string; AShortCut: TShortCut;
      OnExecute: TNotifyEvent; const IcoName: string;
      const AHint: string = ''; UseDefaultIcon: Boolean = False): TCnWizMenuAction;
    {* ����������һ�����˵��� CnWizards Action ����ͬʱ�������ӵ��б��С�
       ʹ�� Add �����Ķ���Ӧ���� Delete �������ͷš�
     |<PRE>
       ACommand: string         - Action �����֣�����Ϊһ��Ψһ���ַ���ֵ
       ACaption: string         - Action �ı���
       AMenuName: string        - �˵�������֣�����Ϊһ��Ψһ���ַ���ֵ�������� ACommand ��ͬ
       AShortCut: TShortCut     - Action ��Ĭ�Ͽ�ݼ���ʵ��ʹ�õļ�ֵ���ע����ж�ȡ
       OnExecute: TNotifyEvent  - ִ��֪ͨ�¼�
       IcoName: string          - Action ������ͼ������֣�����ʱ���Զ�����Դ���ļ��в���װ��
       AHint: string            - Action ����ʾ��Ϣ
       UseDefaultIcon: Boolean  - Action �Ҳ���ͼ��ʱ�Ƿ�ʹ��Ĭ��ͼ��
       Result: TCnWizMenuAction - ���ؽ��Ϊһ�� TCnWizMenuAction ʵ��
     |</PRE>}
    procedure Delete(Index: Integer);
    {* ɾ��ָ�������ŵ� Action ������� Add��}
    procedure DeleteAction(var AWizAction: TCnWizAction);
    {* ɾ��ָ���� Action ������ɺ󽫶��������Ϊ nil����� Add��
       �� Add ������ TCnWizMenuAction ����Ҳʹ�ø÷�����ɾ����}
    procedure Clear;
    {* ������е� Action ���󣬰��� TCnWizAction �� TCnWizMenuAction ��ʵ����}
    function IndexOfAction(AWizAction: TCnWizAction): Integer;
    {* ����ָ���� Action �������б��е������ţ����������� TCnWizAction ��
       TCnWizMenuAction ���󡣷��ص�������ֻ���� WizActions ���������ʹ�á�}
    function IndexOfCommand(const ACommand: string): Integer;
    {* ���� Action ���������������ţ����ص�������ֻ���� WizActions ���������ʹ�á�}
    function IndexOfShortCut(AShortCut: TShortCut): Integer; 
    {* ���ݿ�ݼ���ֵ���������ţ����ص�������ֻ���� WizActions ���������ʹ�á�}
    procedure ArrangeMenuItems(RootItem: TMenuItem; MaxItems: Integer = 0);
    {* Ϊ�����Ĳ˵����ӷָ��˵��� }

    property IdeActionCount: Integer read GetIdeActionCount;
    {* ���� IDE ���� ActionList ��������Ŀ��}
    property WizActionCount: Integer read GetWizActionCount;
    {* �������б��� TCnWizAction ����� TCnWizMenuAction ���������}
    property WizMenuActionCount: Integer read GetWizMenuActionCount;
    {* �������б��� TCnWizMenuAction �������Ŀ��}
    property IdeActions[Index: Integer]: TContainedAction read GetIdeActions;
    {* ���� IDE ���� ActionList ������ Action ���飬������ WizActions ��
       WizMenuActions �������Ķ���}
    property WizActions[Index: Integer]: TCnWizAction read GetWizActions;
    {* �������б��� TCnWizAction ������������飬Ҳ������ TCnWizMenuAction ����}
    property WizMenuActions[Index: Integer]: TCnWizMenuAction read GetWizMenuActions;
    {* �������б��� TCnWizMenuAction ��������}
    property MoreAction: TAction read FMoreAction;
    {* ���ڽ���˵����������õķָ��˵� Action }
  end;

function WizActionMgr: TCnWizActionMgr;
{* ���ص�ǰ�� IDE Action ������ʵ���������Ҫʹ�� IDE Action ���������벻Ҫֱ��
   ���� TCnWizActionMgr ��ʵ������Ӧ���øú��������ʡ�}

procedure FreeWizActionMgr;
{* �ͷŹ�����ʵ��}

implementation

uses
{$IFDEF DEBUG}
  CnDebug,
{$ENDIF}
  CnWizUtils, CnWizIdeUtils;

const
  csUpdateInterval = 100;

//==============================================================================
// CnWizards IDE Action ��װ��
//==============================================================================

{ TCnWizAction }

// �๹����
constructor TCnWizAction.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FCommand := '';
  FIcon := TIcon.Create;
  FWizShortCut := nil;
  FUpdating := False;
end;

// ��������
destructor TCnWizAction.Destroy;
begin
  if Assigned(FWizShortCut) then
    WizShortCutMgr.DeleteShortCut(FWizShortCut);
  FIcon.Free;
  inherited Destroy;
end;

// ���� Action ״̬
function TCnWizAction.Update: Boolean;
begin
  if GetTickCount - FLastUpdateTick > csUpdateInterval then
  begin
    Result := inherited Update;
    FLastUpdateTick := GetTickCount;
  end
  else
    Result := True;  
end;

// ���Ա��֪ͨ
procedure TCnWizAction.Change;
var
  NotifyEvent: TNotifyEvent;
begin
  if FUpdating then Exit;

  if Assigned(FWizShortCut) then
  begin
    // ��ֹ�̳����Ŀ�ݼ����޸�
    if inherited ShortCut <> ShortCut then
    begin
      SetInheritedShortCut;
      Exit;
    end;
  end;

  inherited Change;
  NotifyEvent := OnShortCut;
  if Assigned(FWizShortCut) then
    FWizShortCut.KeyProc := OnShortCut;
end;

// ��ݼ����ù���
procedure TCnWizAction.OnShortCut(Sender: TObject);
begin
  if Assigned(OnExecute) then
    OnExecute(Self);
end;

// ���ô� TAction �̳����� ShortCut ����
procedure TCnWizAction.SetInheritedShortCut;
begin
  Assert(Assigned(FWizShortCut));
  FUpdating := True;
  try
    inherited ShortCut := FWizShortCut.ShortCut;
  finally
    FUpdating := False;
  end;
end;

// ShortCut ���Զ�����
function TCnWizAction.GetShortCut: TShortCut;
begin
  Assert(Assigned(FWizShortCut));
  Result := FWizShortCut.ShortCut;
end;

// ShortCut ����д����
procedure TCnWizAction.{$IFDEF DelphiXE3_UP}_CnSetShortCut{$ELSE}SetShortCut{$ENDIF}(const Value: TShortCut);
begin
  Assert(Assigned(FWizShortCut));
  if FWizShortCut.ShortCut <> Value then
  begin
    FWizShortCut.ShortCut := Value;
    SetInheritedShortCut;
  end;
end;

//==============================================================================
// ���˵���� CnWizards IDE Action ��װ��
//==============================================================================

{ TCnWizMenuAction }

// �๹����
constructor TCnWizMenuAction.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FMenu := nil;
end;

// ��������
destructor TCnWizMenuAction.Destroy;
begin
  if Assigned(FMenu) then
    FreeAndNil(FMenu);
  inherited Destroy;
end;

{ TCnWizActionMgr }

//==============================================================================
// CnWizards IDE Action ��������
//==============================================================================

// �๹����
constructor TCnWizActionMgr.Create(AOwner: TComponent);
begin
{$IFDEF DEBUG}
  CnDebugger.LogEnter('TCnWizActionMgr.Create');
{$ENDIF}
  inherited Create(AOwner);
  FWizActions := TList.Create;
  FWizMenuActions := TList.Create;
  FMoreAction := TAction.Create(nil);
  FMoreAction.Caption := SCnMoreMenu;
  FMoreAction.Hint := StripHotkey(SCnMoreMenu);
  FMoreAction.OnExecute := MoreActionExecute;
{$IFDEF DEBUG}
  CnDebugger.LogLeave('TCnWizActionMgr.Create');
{$ENDIF}
end;

// ��������
destructor TCnWizActionMgr.Destroy;
begin
{$IFDEF DEBUG}
  CnDebugger.LogEnter('TCnWizActionMgr.Destroy');
{$ENDIF}
  Clear;
  FMoreAction.Free;
  FWizActions.Free;
  FWizMenuActions.Free;
  inherited Destroy;
{$IFDEF DEBUG}
  CnDebugger.LogLeave('TCnWizActionMgr.Destroy');
{$ENDIF}
end;

// �Ӷ����ͷ�֪ͨ
procedure TCnWizActionMgr.Notification(AComponent: TComponent;
  Operation: TOperation);
var
  i: Integer;
begin
  inherited;
  if FDeleting then Exit;
  
{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnWizActionMgr.Notification: (%s: %s)',
    [AComponent.Name, AComponent.ClassName]);
{$ENDIF}
  for i := 0 to FWizActions.Count - 1 do
    if FWizActions[i] = AComponent then
    begin
      FWizActions.Delete(i);
      {$IFDEF DEBUG}
        CnDebugger.LogMsg('TCnWizActionMgr FWizActions.Delete.');
      {$ENDIF}
      Exit;
    end;

  for i := 0 to FWizMenuActions.Count - 1 do
    if FWizMenuActions[i] = AComponent then
    begin
      FWizMenuActions.Delete(i);
      {$IFDEF DEBUG}
        CnDebugger.LogMsg('TCnWizActionMgr FWizMenuActions.Delete.');
      {$ENDIF}
      Exit;
    end
    else if TCnWizMenuAction(FWizMenuActions[i]).FMenu = AComponent then
    begin
      TCnWizMenuAction(FWizMenuActions[i]).FMenu := nil;
      Exit;
    end;
end;

//------------------------------------------------------------------------------
// �б���Ŀ����
//------------------------------------------------------------------------------

// ��ʼ�� Action ����
procedure TCnWizActionMgr.InitAction(AWizAction: TCnWizAction;
  const ACommand, ACaption: string; OnExecute: TNotifyEvent; OnUpdate: TNotifyEvent;
  const IcoName, AHint: string; UseDefaultIcon: Boolean);
var
  Svcs40: INTAServices40;
  NewName: string;
begin
  QuerySvcs(BorlandIDEServices, INTAServices40, Svcs40);
  if Trim(ACommand) <> '' then
  begin
    NewName := SCnActionPrefix + Trim(ACommand);
    if Svcs40.ActionList.FindComponent(NewName) = nil then
    begin
      try
        AWizAction.Name := NewName;
      except
      {$IFDEF DEBUG}
        CnDebugger.LogMsgWithType('Rename action error: ' + NewName, cmtError);
      {$ENDIF}
      end;
    end
    else
    {$IFDEF DEBUG}
      CnDebugger.LogMsgWithType('Component is already exists: ' + NewName, cmtError);
    {$ENDIF}
  end;
  AWizAction.Caption := ACaption;
  AWizAction.Hint := AHint;
  AWizAction.Category := SCnWizardsActionCategory;
  AWizAction.OnExecute := OnExecute;
  AWizAction.OnUpdate := OnUpdate;
  
  AWizAction.ActionList := Svcs40.ActionList;
  if CnWizLoadIcon(nil, AWizAction.FIcon, IcoName, UseDefaultIcon) then
  begin
{$IFDEF DEBUG}
    CnDebugger.LogFmt('Load Icon %s OK with %dx%d', [IcoName, AWizAction.FIcon.Width,
      AWizAction.FIcon.Height]);
{$ENDIF}

{$IFDEF IDE_SUPPORT_HDPI}
    AWizAction.ImageIndex := AddGraphicToVirtualImageList(AWizAction.FIcon, Svcs40.ImageList as TVirtualImageList)
{$ELSE}
    AWizAction.ImageIndex := AddIconToImageList(AWizAction.FIcon, Svcs40.ImageList, False)
{$ENDIF}
  end
  else
    AWizAction.ImageIndex := -1;
  AWizAction.FCommand := ACommand;
end;

// ����������һ�����˵��� CnWizards Action ����ͬʱ�������ӵ��б���
function TCnWizActionMgr.AddMenuAction(const ACommand, ACaption, AMenuName: string;
  AShortCut: TShortCut; OnExecute: TNotifyEvent; const IcoName,
  AHint: string; UseDefaultIcon: Boolean): TCnWizMenuAction;
var
  Svcs40: INTAServices40;
begin
{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnWizActionMgr.Add WizMenuAction: %s', [ACommand]);
{$ENDIF}
  if IndexOfCommand(ACommand) >= 0 then
    raise ECnDuplicateCommand.CreateFmt(SCnDuplicateCommand, [ACommand]);
    
  QuerySvcs(BorlandIDEServices, INTAServices40, Svcs40);
  Result := TCnWizMenuAction.Create(Svcs40.ActionList);
  Result.FreeNotification(Self);

  Result.FUpdating := True;         // ��ʼ����
  try
    InitAction(Result, ACommand, ACaption, OnExecute, nil, IcoName, AHint, UseDefaultIcon);
    Result.FMenu := TMenuItem.Create(nil);
    Result.FMenu.FreeNotification(Self);
    Result.FMenu.Name := AMenuName;
    Result.FMenu.Action := Result;
    Result.FMenu.AutoHotkeys := maManual;
    Result.FWizShortCut := WizShortCutMgr.Add(ACommand, AShortCut, Result.OnShortCut,
      AMenuName, 0, Result);

    Result.SetInheritedShortCut;
    FWizMenuActions.Add(Result);
  finally
    Result.FUpdating := False;
  end;
{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnWizActionMgr.Add WizMenuAction: %s Complete.', [ACommand]);
{$ENDIF}
end;

// ����������һ�� CnWizards Action ����ͬʱ�������ӵ��б���
function TCnWizActionMgr.AddAction(const ACommand, ACaption: string;
  AShortCut: TShortCut; OnExecute: TNotifyEvent; const IcoName,
  AHint: string; UseDefaultIcon: Boolean): TCnWizAction;
var
  Svcs40: INTAServices40;
begin
{$IFDEF DEBUG}
  CnDebugger.LogFmt('TCnWizActionMgr.Add WizAction: %s', [ACommand]);
{$ENDIF}
  if IndexOfCommand(ACommand) >= 0 then
    raise ECnDuplicateCommand.CreateFmt(SCnDuplicateCommand, [ACommand]);
    
  QuerySvcs(BorlandIDEServices, INTAServices40, Svcs40);
  Result := TCnWizAction.Create(Svcs40.ActionList);
  Result.FreeNotification(Self);

  Result.FUpdating := True;         // ��ʼ����
  try
    InitAction(Result, ACommand, ACaption, OnExecute, nil, IcoName, AHint, UseDefaultIcon);
    Result.FWizShortCut := WizShortCutMgr.Add(ACommand, AShortCut, Result.OnShortCut, '', 0, Result);
    Result.SetInheritedShortCut;
    FWizActions.Add(Result);
  finally
    Result.FUpdating := False;
  end;
end;

// ����б�
procedure TCnWizActionMgr.Clear;
begin
{$IFDEF DEBUG}
  CnDebugger.LogEnter('TCnWizActionMgr.Clear');
{$ENDIF}
  WizShortCutMgr.BeginUpdate;       // ��ʼ����
  try
    while WizActionCount > 0 do
      Delete(0);
  finally
    WizShortCutMgr.EndUpdate;       // �������£����°󶨿�ݼ�
  end;
{$IFDEF DEBUG}
  CnDebugger.LogLeave('TCnWizActionMgr.Clear');
{$ENDIF}
end;

// ɾ��ָ�������ŵ� Action ����
procedure TCnWizActionMgr.Delete(Index: Integer);
begin
  FDeleting := True;
  try
    if (Index >= 0) and (Index < FWizActions.Count) then
    begin
    {$IFDEF DEBUG}
      CnDebugger.LogFmt('TCnWizActionMgr.Delete(%d Action): %s', [Index,
        TCnWizAction(FWizActions[Index]).Command]);
    {$ENDIF}
      TCnWizAction(FWizActions[Index]).Free;
      FWizActions.Delete(Index);
    end
    else if (Index >= FWizActions.Count) and (Index < FWizActions.Count +
      FWizMenuActions.Count) then
    begin
    {$IFDEF DEBUG}
      CnDebugger.LogFmt('TCnWizActionMgr.Delete(%d MenuAction): %s', [Index,
        TCnWizAction(FWizMenuActions[Index - FWizActions.Count]).Command]);
    {$ENDIF}
      TCnWizMenuAction(FWizMenuActions[Index - FWizActions.Count]).Free;
      FWizMenuActions.Delete(Index - FWizActions.Count);
    end;
  finally
    FDeleting := False;
  end;
end;

// ɾ��ָ���� Action ������ɺ󽫶��������Ϊ nil
procedure TCnWizActionMgr.DeleteAction(var AWizAction: TCnWizAction);
begin
  Delete(IndexOfAction(AWizAction));
  AWizAction := nil;
end;

// ����ָ���� Action �������б��е�������
function TCnWizActionMgr.IndexOfAction(AWizAction: TCnWizAction): Integer;
var
  I: Integer;
begin
  Result := -1;
  for I := 0 to WizActionCount - 1 do
  begin
    if AWizAction = WizActions[I] then
    begin
      Result := I;
      Exit;
    end;
  end;
end;

// ���� Action ����������������
function TCnWizActionMgr.IndexOfCommand(const ACommand: string): Integer;
var
  I: Integer;
begin
  Result := -1;
  if ACommand = '' then Exit;
  for I := 0 to WizActionCount - 1 do
    if WizActions[I].FCommand = ACommand then
    begin
      Result := I;
      Exit;
    end;
end;

// ���ݿ�ݼ���ֵ����������
function TCnWizActionMgr.IndexOfShortCut(AShortCut: TShortCut): Integer;
var
  I: Integer;
begin
  Result := -1;
  if AShortCut = 0 then Exit;
  for I := 0 to WizActionCount - 1 do
    if WizActions[I].ShortCut = AShortCut then
    begin
      Result := I;
      Exit;
    end;
end;

procedure TCnWizActionMgr.MoreActionExecute(Sender: TObject);
begin
  // do nothing
end;

procedure TCnWizActionMgr.ArrangeMenuItems(RootItem: TMenuItem;
  MaxItems: Integer);
{$IFDEF COMPILER7_UP}
  function NewMoreItem: TMenuItem;
  begin
    Result := TMenuItem.Create(RootItem);
    Result.Action := MoreAction;
  end;

var
  I: Integer;
  ScreenRect: TRect;
  ScreenHeight: Integer;
  MoreMenuItem: TMenuItem;
  ParentItem: TMenuItem;
  Item: TMenuItem;
  CurrentIndex: Integer;
  MenuItems: array of TMenuItem;
{$ENDIF}
begin
{$IFDEF COMPILER7_UP}
  if MaxItems < 8 then
  begin
    ScreenRect := GetWorkRect(GetIdeMainForm);
    ScreenHeight := ScreenRect.Bottom - ScreenRect.Top - 75;
    MaxItems := ScreenHeight div GetMainMenuItemHeight;
    if MaxItems < 8 then MaxItems := 8;
  end;

{$IFDEF DEBUG}
  CnDebugger.LogMsg('CnWizActionMgr.ArrangeMenuItems with Max ' + IntToStr(MaxItems));
{$ENDIF}

  SetLength(MenuItems, RootItem.Count);
  try
    for I := RootItem.Count - 1 downto 0 do
      MenuItems[I] := RootItem.Items[I];

    ParentItem := RootItem;
    CurrentIndex := 0;

    for I := 0 to Length(MenuItems) - 1 do
    begin
      Item := MenuItems[I];
      if (CurrentIndex = MaxItems - 1) and (I < (Length(MenuItems) - 1)) then
      begin
        MoreMenuItem := NewMoreItem;
        ParentItem.Add(MoreMenuItem);
{$IFDEF DEBUG}
        CnDebugger.LogMsg('CnWizActionMgr.ArrangeMenuItems. Add MoreItem at ' + IntToStr(CurrentIndex));
{$ENDIF}
        ParentItem := MoreMenuItem;
        CurrentIndex := 0;
      end;
      if Item.Parent <> ParentItem then
      begin
        Item.Parent.Remove(Item);
        ParentItem.Add(Item);
      end;
      Item.MenuIndex := CurrentIndex;
      Inc(CurrentIndex);
    end;
  finally
    MenuItems := nil;
  end;          
{$ENDIF}
end;

//------------------------------------------------------------------------------
// ���Զ�д����
//------------------------------------------------------------------------------

// IdeActionCount ���Զ�����
function TCnWizActionMgr.GetIdeActionCount: Integer;
var
  Svcs40: INTAServices40;
begin
  QuerySvcs(BorlandIDEServices, INTAServices40, Svcs40);
  Result := Svcs40.ActionList.ActionCount;
end;

// IdeActions ���Զ�����
function TCnWizActionMgr.GetIdeActions(Index: Integer): TContainedAction;
var
  Svcs40: INTAServices40;
begin
  QuerySvcs(BorlandIDEServices, INTAServices40, Svcs40);
  Result := Svcs40.ActionList.Actions[Index];
end;

// WizActionCount ���Զ�����
function TCnWizActionMgr.GetWizActionCount: Integer;
begin
  Result := FWizActions.Count + FWizMenuActions.Count;
end;

// WizActions ���Զ�����
function TCnWizActionMgr.GetWizActions(Index: Integer): TCnWizAction;
begin
  if (Index >= 0) and (Index < FWizActions.Count) then
    Result := TCnWizAction(FWizActions[Index])
  else if (Index >= FWizActions.Count) and (Index < FWizActions.Count +
    FWizMenuActions.Count) then
    Result := TCnWizAction(FWizMenuActions[Index - FWizActions.Count])
  else
    Result := nil;
end;

// WizMenuActionCount ���Զ�����
function TCnWizActionMgr.GetWizMenuActionCount: Integer;
begin
  Result := FWizMenuActions.Count;
end;

// WizMenuActions ���Զ�����
function TCnWizActionMgr.GetWizMenuActions(
  Index: Integer): TCnWizMenuAction;
begin
  Result := nil;
  if (Index >= 0) and (Index < FWizMenuActions.Count) then
    Result := TCnWizMenuAction(FWizMenuActions[Index]);
end;

var
  FWizActionMgr: TCnWizActionMgr = nil;

// ���ص�ǰ�� IDE Action ������ʵ��
function WizActionMgr: TCnWizActionMgr;
begin
  if FWizActionMgr = nil then
    FWizActionMgr := TCnWizActionMgr.Create(nil);
  Result := FWizActionMgr;
end;

// �ͷŹ�����ʵ��
procedure FreeWizActionMgr;
begin
  if FWizActionMgr <> nil then
    FreeAndNil(FWizActionMgr);
end;

end.


