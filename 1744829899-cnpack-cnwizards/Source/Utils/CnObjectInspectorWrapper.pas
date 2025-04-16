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

unit CnObjectInspectorWrapper;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�����鿴���Ĳ�����װ��Ԫ
* ��Ԫ���ߣ�CnPack ������
* ��    ע��ע�����鿴�����ܴ����ϳ٣��÷�װ����ֵ��������ж�
* ����ƽ̨��Win7 + Delphi 5.01
* ���ݲ��ԣ�Win7 + D5/2007/2009
* �� �� �����ô����е��ַ����ݲ�֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2025.02.02 V1.2
*               ���Ӷ���鿴���ͷ�ʱͬ���ͷ����ü� Hook �Ļ��Ʊ����˳�ʱ����
*           2025.01.31 V1.1
*               ���Ӷ���鿴��������֪ͨ
*           2025.01.05 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  SysUtils, Classes, Controls, Forms, TypInfo, Menus, CnEventHook;

type
  TCnObjectInspectorWrapper = class(TComponent)
  {* ����鿴���ķ�װ}
  private
    FObjectInspectorForm: TCustomForm;  // ����鿴������
    FPropListBox: TControl;             // ����鿴���ڲ��б�
    FTabControl: TControl;              // ����鿴�������¼� Tab
    FPopupMenu: TPopupMenu;             // ����鿴���Ҽ��˵�
    FListEventHook: TCnEventHook;       // �ҽ������б�ѡ��ı��¼�
    FTabEventHook: TCnEventHook;        // �ҽ������¼� Tab �л��¼�
    FSelectionChangeNotifiers: TList;
    FObjectInspectorCreatedNotifiers: TList;
    function GetActiveComponentName: string;
    function GetActiveComponentType: string;
    function GetActivePropName: string;
    function GetActivePropValue: string;
    function GetShowGridLines: Boolean;
    procedure SetShowGridLines(const Value: Boolean);
  protected
    procedure ActiveFormChanged(Sender: TObject);
    procedure SelectionItem(Sender: TObject);
    procedure TabChange(Sender: TObject);
    procedure CheckObjectInspector;
    procedure InitObjectInspector;

    procedure Notification(AComponent: TComponent; Operation: TOperation); override;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

    procedure AddSelectionChangeNotifier(Notifier: TNotifyEvent);
    {* ����һ��ѡ�иı��֪ͨ}
    procedure RemoveSelectionChangeNotifier(Notifier: TNotifyEvent);
    {* ɾ��һ��ѡ�иı��֪ͨ}

    procedure AddObjectInspectorCreatedNotifier(Notifier: TNotifyEvent);
    {* ����һ������鿴��������֪ͨ��Դ�ڶ���鿴������ IDE ������ĺܾ��Ժ�Ŵ���}
    procedure RemoveObjectInspectorCreatedNotifier(Notifier: TNotifyEvent);
    {* ɾ��һ������鿴��������֪ͨ}

    procedure RepaintPropList;
    {* �ػ��б�}

    property ActiveComponentType: string read GetActiveComponentType;
    {* ����鿴����ǰѡ�е����������ѡ�ж�����ʱΪ�գ����������ͬ��}
    property ActiveComponentName: string read GetActiveComponentName;
    {* ����鿴����ǰѡ�е��������ѡ�ж�����ʱΪ 2 items selected ����}
    property ActivePropName: string read GetActivePropName;
    {* ����鿴����ǰѡ�е�������}
    property ActivePropValue: string read GetActivePropValue;
    {* ����鿴����ǰѡ�е�����ֵ}
    property ShowGridLines: Boolean read GetShowGridLines write SetShowGridLines;
    {* ����鿴����ʾ���Ƿ���ʾ���񣬽� Delphi 6 �����ϰ汾��Ч}

    property PopupMenu: TPopupMenu read FPopupMenu;
    {* �����Ĳ˵�}
  end;

function ObjectInspectorWrapper: TCnObjectInspectorWrapper;
{* ��ȡȫ�ֶ���鿴���ķ�װ����}

implementation

uses
  CnWizIdeUtils, CnWizNotifier, CnWizUtils {$IFDEF DEBUG}, CnDebug {$ENDIF};

type
  TNotifyEventProc = procedure (Self: TObject; Sender: TObject);

var
  FObjectInspectorWrapper: TCnObjectInspectorWrapper = nil;

function ObjectInspectorWrapper: TCnObjectInspectorWrapper;
begin
  if FObjectInspectorWrapper = nil then
    FObjectInspectorWrapper := TCnObjectInspectorWrapper.Create(nil);
  Result := FObjectInspectorWrapper;
end;

{ TCnObjectInspectorWrapper }

procedure TCnObjectInspectorWrapper.AddSelectionChangeNotifier(
  Notifier: TNotifyEvent);
begin
  CnWizAddNotifier(FSelectionChangeNotifiers, TMethod(Notifier));
end;

procedure TCnObjectInspectorWrapper.RemoveSelectionChangeNotifier(
  Notifier: TNotifyEvent);
begin
  CnWizRemoveNotifier(FSelectionChangeNotifiers, TMethod(Notifier));
end;

procedure TCnObjectInspectorWrapper.AddObjectInspectorCreatedNotifier(
  Notifier: TNotifyEvent);
begin
  CnWizAddNotifier(FObjectInspectorCreatedNotifiers, TMethod(Notifier));
end;

procedure TCnObjectInspectorWrapper.RemoveObjectInspectorCreatedNotifier(
  Notifier: TNotifyEvent);
begin
  CnWizRemoveNotifier(FObjectInspectorCreatedNotifiers, TMethod(Notifier));
end;

procedure TCnObjectInspectorWrapper.CheckObjectInspector;
begin
  if (FObjectInspectorForm = nil) or (FPropListBox = nil) then
    InitObjectInspector;
end;

constructor TCnObjectInspectorWrapper.Create(AOwner: TComponent);
begin
  inherited;

  FSelectionChangeNotifiers := TList.Create;
  FObjectInspectorCreatedNotifiers := TList.Create;

  InitObjectInspector;

  CnWizNotifierServices.AddActiveFormNotifier(ActiveFormChanged);
end;

destructor TCnObjectInspectorWrapper.Destroy;
begin
  if FObjectInspectorForm <> nil then
    FObjectInspectorForm.RemoveFreeNotification(Self);
  if FPropListBox <> nil then
    FPropListBox.RemoveFreeNotification(Self);
  if FTabControl <> nil then
    FTabControl.RemoveFreeNotification(Self);
  if FPopupMenu <> nil then
    FPopupMenu.RemoveFreeNotification(Self);

  CnWizNotifierServices.RemoveActiveFormNotifier(ActiveFormChanged);

  FTabEventHook.Free;
  FListEventHook.Free;
  CnWizClearAndFreeList(FObjectInspectorCreatedNotifiers);
  CnWizClearAndFreeList(FSelectionChangeNotifiers);
  inherited;
end;

function TCnObjectInspectorWrapper.GetActiveComponentName: string;
begin
  CheckObjectInspector;
  if FObjectInspectorForm <> nil then
    Result := GetStrProp(FObjectInspectorForm, 'ActiveComponentName')
  else
    Result := '';
end;

function TCnObjectInspectorWrapper.GetActiveComponentType: string;
begin
  CheckObjectInspector;
  if FObjectInspectorForm <> nil then
    Result := GetStrProp(FObjectInspectorForm, 'ActiveComponentType')
  else
    Result := '';
end;

function TCnObjectInspectorWrapper.GetActivePropName: string;
begin
  CheckObjectInspector;
  if FObjectInspectorForm <> nil then
    Result := GetStrProp(FObjectInspectorForm, 'ActivePropName')
  else
    Result := '';
end;

function TCnObjectInspectorWrapper.GetActivePropValue: string;
begin
  CheckObjectInspector;
  if FObjectInspectorForm <> nil then
    Result := GetStrProp(FObjectInspectorForm, 'ActivePropValue')
  else
    Result := '';
end;

function TCnObjectInspectorWrapper.GetShowGridLines: Boolean;
var
  PropInfo: PPropInfo;
begin
  CheckObjectInspector;
  Result := False;

  if FPropListBox <> nil then
  begin
    try
      PropInfo := GetPropInfo(FPropListBox, 'ShowGridLines');
      if PropInfo <> nil then
        Result := GetOrdProp(FPropListBox, 'ShowGridLines') <> 0;
    except
      ; // ��һ FreeNotification û�����ã����� IDE �ر��ڼ����鿴���ͷ��˵� FPropListBox �������������˴���ʱ����
    end;
  end;
end;

procedure TCnObjectInspectorWrapper.InitObjectInspector;
var
  C: TComponent;
{$IFDEF DEBUG}
  PropInfo: PPropInfo;
{$ENDIF}
begin
  // �Ҵ���
  FObjectInspectorForm := GetObjectInspectorForm;
  if FObjectInspectorForm <> nil then
  begin
    FObjectInspectorForm.FreeNotification(Self);
{$IFDEF DEBUG}
    PropInfo := GetPropInfo(FObjectInspectorForm, 'ActiveComponentType');
    if PropInfo <> nil then
      CnDebugger.LogMsg('TCnObjectInspectorWrapper ActiveComponentType ' + PropInfo^.PropType^.Name);

    PropInfo := GetPropInfo(FObjectInspectorForm, 'ActiveComponentName');
    if PropInfo <> nil then
      CnDebugger.LogMsg('TCnObjectInspectorWrapper ActiveComponentName ' + PropInfo^.PropType^.Name);

    PropInfo := GetPropInfo(FObjectInspectorForm, 'ActivePropName');
    if PropInfo <> nil then
      CnDebugger.LogMsg('TCnObjectInspectorWrapper ActivePropName ' + PropInfo^.PropType^.Name);

    PropInfo := GetPropInfo(FObjectInspectorForm, 'ActivePropValue');
    if PropInfo <> nil then
      CnDebugger.LogMsg('TCnObjectInspectorWrapper ActivePropValue ' + PropInfo^.PropType^.Name);
{$ENDIF}

    C := FObjectInspectorForm.FindComponent(SCnPropertyInspectorListName);
    if C <> nil then
    begin
      if C is TControl then
      begin
        FPropListBox := TControl(C);
        FPropListBox.FreeNotification(Self);

{$IFDEF DEBUG}
        PropInfo := GetPropInfo(FPropListBox, 'ShowGridLines');
        if PropInfo <> nil then
          CnDebugger.LogMsg('TCnObjectInspectorWrapper ShowGridLines ' + PropInfo^.PropType^.Name)
        else
          CnDebugger.LogMsg('TCnObjectInspectorWrapper ShowGridLines NOT Exists.');
{$ENDIF}

        // Hook �� Selection Change �¼�
        FListEventHook := TCnEventHook.Create(FPropListBox, 'OnSelectItem',
          Self, @TCnObjectInspectorWrapper.SelectionItem);
        // ע��˴�Ӧ�� Self��ȷ�� SelectionItem �����õ� Self ����ȷ��

{$IFDEF DEBUG}
        CnDebugger.LogMsg('TCnObjectInspectorWrapper.InitObjectInspector List Hooked.');
{$ENDIF}
      end;
    end;

    // �� TabControl������������ TTXTabControl �� TCodeEditorTabControl �������ɴ಻�ж� 
    C := FObjectInspectorForm.FindComponent(SCnPropertyInspectorTabControlName);
    if C <> nil then
    begin
      if C is TControl then
      begin
        FTabControl := TControl(C);
        FTabControl.FreeNotification(Self);

        // Hook �� Change �¼�
        FTabEventHook := TCnEventHook.Create(FTabControl, 'OnChange',
          Self, @TCnObjectInspectorWrapper.TabChange);
        // ע��˴�Ӧ�� Self��ȷ�� TabChange �����õ� Self ����ȷ��

{$IFDEF DEBUG}
        CnDebugger.LogMsg('TCnObjectInspectorWrapper.InitObjectInspector Tab Hooked.');
{$ENDIF}
      end;
    end;

    // ���Ҽ��˵�
    C := FObjectInspectorForm.FindComponent(SCnPropertyInspectorLocalPopupMenu);
    if C <> nil then
    begin
      if C is TPopupMenu then
      begin
        FPopupMenu := TPopupMenu(C);
        FPopupMenu.FreeNotification(Self);
      end;
    end;
  end;
end;

procedure TCnObjectInspectorWrapper.RepaintPropList;
begin
  if FPropListBox <> nil then
    FPropListBox.Repaint;
end;

procedure TCnObjectInspectorWrapper.SelectionItem(Sender: TObject);
var
  I: Integer;
begin
  if FListEventHook.Trampoline <> nil then
    TNotifyEventProc(FListEventHook.Trampoline)(FListEventHook.TrampolineData, Sender);

  // ������ɴ����¼��󣬷���֪ͨ
  if FSelectionChangeNotifiers <> nil then
  begin
    for I := FSelectionChangeNotifiers.Count - 1 downto 0 do
    try
      with PCnWizNotifierRecord(FSelectionChangeNotifiers[I])^ do
        TNotifyEvent(Notifier)(Sender);
    except
      DoHandleException('TCnObjectInspectorWrapper.SelectionItem[' + IntToStr(I) + ']');
    end;
  end;
end;

procedure TCnObjectInspectorWrapper.TabChange(Sender: TObject);
var
  I: Integer;
begin
  if FTabEventHook.Trampoline <> nil then
    TNotifyEventProc(FTabEventHook.Trampoline)(FTabEventHook.TrampolineData, Sender);

  // ������ɴ����¼��󣬷���֪ͨ
  if FSelectionChangeNotifiers <> nil then
  begin
    for I := FSelectionChangeNotifiers.Count - 1 downto 0 do
    try
      with PCnWizNotifierRecord(FSelectionChangeNotifiers[I])^ do
        TNotifyEvent(Notifier)(Sender);
    except
      DoHandleException('TCnObjectInspectorWrapper.TabChange[' + IntToStr(I) + ']');
    end;
  end;
end;

procedure TCnObjectInspectorWrapper.SetShowGridLines(const Value: Boolean);
var
  PropInfo: PPropInfo;
begin
  CheckObjectInspector;
  if FPropListBox <> nil then
  begin
    try
      PropInfo := GetPropInfo(FPropListBox, 'ShowGridLines');
      if PropInfo <> nil then
      begin
        SetOrdProp(FPropListBox, 'ShowGridLines', Ord(Value));
        FPropListBox.Repaint;
      end;
    except
      ; // Ҳ���Σ�����һ
    end;
  end;
end;

procedure TCnObjectInspectorWrapper.ActiveFormChanged(Sender: TObject);
var
  I: Integer;
  IsNil: Boolean;
begin
  IsNil := FObjectInspectorForm = nil;
  CheckObjectInspector;
  if FObjectInspectorForm <> nil then
  begin
    // ֪ͨ��⵽�˶���鿴���Ĵ���
    if FObjectInspectorCreatedNotifiers <> nil then
    begin
      for I := FObjectInspectorCreatedNotifiers.Count - 1 downto 0 do
      try
        with PCnWizNotifierRecord(FObjectInspectorCreatedNotifiers[I])^ do
          TNotifyEvent(Notifier)(Sender);
      except
        DoHandleException('TCnObjectInspectorWrapper.ObjectInspectorCreated[' + IntToStr(I) + ']');
      end;
    end;
  end;
end;

procedure TCnObjectInspectorWrapper.Notification(AComponent: TComponent;
  Operation: TOperation);
begin
  inherited;
  if (AComponent = FObjectInspectorForm) or (AComponent = FPropListBox)
    or (AComponent = FTabControl) or (AComponent = FPopupMenu) then
  begin
{$IFDEF DEBUG}
    CnDebugger.LogMsg('Object Inspector Form Free Notification. Set Nil and UnHook.');
{$ENDIF}

    FObjectInspectorForm := nil;
    FPropListBox := nil;
    FTabControl := nil;
    FPopupMenu := nil;

    FreeAndNil(FListEventHook);
    FreeAndNil(FTabEventHook);
{$IFDEF DEBUG}
    CnDebugger.LogMsg('Object Inspector Form Free Notification. Set Nil and UnHook OK.');
{$ENDIF}
  end;
end;

initialization

finalization
  if FObjectInspectorWrapper <> nil then
    FreeAndNil(FObjectInspectorWrapper);

end.
