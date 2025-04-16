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

unit CnMdiView;
{ |<PRE>
================================================================================
* ������ƣ�CnDebugViewer
* ��Ԫ���ƣ��Ӵ��嵥Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7
* �� �� �����õ�Ԫ�е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2014.10.05
*               ��ʾͳһʹ�� FViewStore �������� FStore ���ܲ�һ�µ����⡣
*           2008.01.18
*               Sesame ���ӷ��뷽ʽ��ʾʱ��
*           2005.01.01
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  CnMsgClasses, ComCtrls, StdCtrls, ExtCtrls, Grids, VirtualTrees, Buttons,
  CnLangMgr, ToolWin, Menus, ActnList;

type
  TCnFilterUpdate = (fuThreadId, fuTag);
  TCnFilterUpdates = set of TCnFilterUpdate;

  TCnMemoContent = (mcNone, mcMsg, mcTime);

  TCnMsgChild = class(TForm)
    pnlTime: TPanel;
    splDetail: TSplitter;
    splTime: TSplitter;
    pnlTree: TPanel;
    pnlFilter: TPanel;
    Splitter3: TSplitter;
    pnlLevel: TPanel;
    cbbLevel: TComboBox;
    Splitter4: TSplitter;
    pnlThread: TPanel;
    cbbThread: TComboBox;
    pnlMsg: TPanel;
    Splitter5: TSplitter;
    pnlType: TPanel;
    cbbType: TComboBox;
    pnlLabel: TPanel;
    pnlTag: TPanel;
    cbbTag: TComboBox;
    pnlDetail: TPanel;
    mmoDetail: TMemo;
    cbbSearch: TComboBox;
    btnSearch: TSpeedButton;
    Label1: TLabel;
    tlbBookmark: TToolBar;
    btnBookmark: TToolButton;
    pmTree: TPopupMenu;
    C1: TMenuItem;
    A1: TMenuItem;
    D1: TMenuItem;
    E1: TMenuItem;
    S1: TMenuItem;
    X1: TMenuItem;
    F1: TMenuItem;
    B1: TMenuItem;
    N1: TMenuItem;
    N2: TMenuItem;
    MenuDropBookmark: TMenuItem;
    M1: TMenuItem;
    SaveMemDump1: TMenuItem;
    CopyText1: TMenuItem;
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormActivate(Sender: TObject);
    procedure HeadPanelResize(Sender: TObject);
    procedure FilterChange(Sender: TObject);
    procedure cbbSearchKeyPress(Sender: TObject; var Key: Char);
    procedure btnSearchClick(Sender: TObject);
    procedure pmTreePopup(Sender: TObject);
    procedure FormShow(Sender: TObject);
  private
    FStore: TCnMsgStore;     // �����ⲿ�߳�������д��֪ͨ���������
    FViewStore: TCnMsgStore; // ������ʾ�ģ����ڲ�Itemֻ�����ã��� FStore ����ͬ������
    FProcessID: DWORD;
    FProcName: string;
    FMsgTree: TVirtualStringTree;
    FTimeTree: TVirtualStringTree;
    FFilter: TCnDisplayFilter;
    FMemContent: TCnMemoContent;
    FSelectedIndex: Integer;
    FBookmarks: array[0..9] of Integer;
    FHasBookmarks: Boolean;
    FIsResizing: Boolean;

    procedure TreeGetText(Sender: TBaseVirtualTree; Node: PVirtualNode;
      Column: TColumnIndex; TextType: TVSTTextType; var CellText: WideString);
    procedure TreeColumnResize(Sender: TVTHeader; Column: TColumnIndex);
    // procedure TreeClick(Sender: TObject);
    procedure TreeChange(Sender: TBaseVirtualTree; Node: PVirtualNode);
    procedure TreeBeforeItemPaint(Sender: TBaseVirtualTree;
      TargetCanvas: TCanvas; Node: PVirtualNode; ItemRect: TRect;
      var CustomDraw: Boolean);
    procedure TreeBeforeItemErase(Sender: TBaseVirtualTree; TargetCanvas:
      TCanvas; Node: PVirtualNode; ItemRect: TRect; var ItemColor: TColor;
      var EraseAction: TItemEraseAction);

    procedure TreeEnter(Sender: TObject);
    procedure TreeKeyPress(Sender: TObject; var Key: Char);

    procedure TimeGetText(Sender: TBaseVirtualTree; Node: PVirtualNode;
      Column: TColumnIndex; TextType: TVSTTextType; var CellText: WideString);
    procedure TimeChange(Sender: TBaseVirtualTree; Node: PVirtualNode);
    procedure TimeEnter(Sender: TObject);

    procedure InitTree;
    procedure InitTimeTree;
    procedure InitControls;

    procedure UpdateToViewStore(Source, Dest: TCnMsgStore);
    {* ���յ�����Ϣ�� FStore ͬ���� FViewStore ��}
    procedure UpdateConditionsToView(Content: TCnFilterUpdates);
    procedure UpdateViewStoreToTree;
    {* �� FViewStore ������ȫ��ͬ�������� Tree �У��ڲ���ѭ������ AddAItemToTree}
    procedure AddAItemToTree(var OldIndent: Integer; var PrevNode: PVirtualNode;
      AItem: TCnMsgItem);
    {* ��������Ϣ��Ŀ��ӵ�����}
    procedure AddBatchItemToView(AStore: TCnMsgStore; StartIndex, EndIndex: Integer);
    {* ����һ����Ϣ��������ӵ� FStore���ڲ���ѭ������ AddAItemToTree}
    procedure UpdateTimeToTree(AStore: TCnMsgStore);
    {* �� AStore ���ȫ����ʱ��Ŀͬ�������� Tree�У��ڲ���ѭ������ AddATimeToTree}
    procedure AddATimeToTree(AItem: TCnTimeItem);
    {* ��������ʱ��Ŀ��ӵ�����}
    procedure RefreshTime(Sender: TObject);

    function GetAnEmptyBookmarkSlot: Integer;
    function GetSlotFromBookmarkLine(Line: Integer): Integer;
    procedure SetSlotToBookmark(Slot, Line: Integer);
    procedure ReleaseAnBookmarkSlot(Slot: Integer);
    procedure ClearBookMarks;
    procedure BookmarkMenuClick(Sender: TObject);
    procedure pnlTreeOnResize(Sender: TObject);
    procedure DoTreeResize;
    procedure SetStore(const Value: TCnMsgStore);
    function GetSelectedContent: string;
    function GetSelectedText: string;
    function GetSelectedItem: TCnMsgItem;
  protected
    procedure DoCreate; override;
    procedure LanguageChanged(Sender: TObject);
  public
    procedure OnStoreChange(Sender: TObject; Operation: TCnStoreChangeType;
      StartIndex, EndIndex: Integer);
    procedure ClearStores;
    procedure ClearTimes;
    procedure LoadFromFile(const FileName: string);
    procedure SaveToFile(const FileName: string);
    procedure FindNode(const AText: string; IsDown: Boolean; IsSeperator: Boolean = False);
    function CheckFind(AItem: TCnMsgItem; const AText: string; IsSeperator: Boolean = False): Boolean;
    procedure ToggleBookmark;
    procedure UpdateBookmarkMenu;
    procedure UpdateBookmarkToMainMenu;
    procedure GotoPrevBookmark;
    procedure GotoNextBookmark;
    procedure ClearAllBookmarks;
    procedure RequireRefreshTime;
    procedure InitFont;
    function DescriptionOfMsg(Index: Integer; AMsgItem: TCnMsgItem): string;
    function DescriptionOfTime(Index: Integer): string;

    property Store: TCnMsgStore read FStore write SetStore;
    property Filter: TCnDisplayFilter read FFilter;
    property ProcessID: DWORD read FProcessID write FProcessID;
    property ProcName: string read FProcName write FProcName;
    property MsgTree: TVirtualStringTree read FMsgTree;
    property TimeTree: TVirtualStringTree read FTimeTree;
    property HasBookmarks: Boolean read FHasBookmarks;
    property IsResizing: Boolean read FIsResizing write FIsResizing;
    property SelectedContent: string read GetSelectedContent;
    property SelectedText: string read GetSelectedText;
    property SelectedItem: TCnMsgItem read GetSelectedItem;
  end;

var
  CnMsgChild: TCnMsgChild;

implementation

uses CnCommon, CnViewMain, CnViewCore, CnDebugIntf, CnMsgFiler;

{$R *.DFM}

procedure TCnMsgChild.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  CnMainViewer.UpdateFormInSwitch(Self, fsDelete);
  Action := caFree;
end;

procedure TCnMsgChild.FormCreate(Sender: TObject);
begin
  // ��ʼ������������ Store
  FViewStore := TCnMsgStore.Create(nil, False);
  FStore := CnMsgManager.AddStore(0, SCnNoneProcName);
  if FStore <> nil then
    FStore.OnChange := OnStoreChange;

  CnMainViewer.UpdateFormInSwitch(Self, fsAdd);
  FMsgTree := TVirtualStringTree.Create(Self);
  FTimeTree := TVirtualStringTree.Create(Self);
  FFilter := TCnDisplayFilter.Create;

  InitControls;
end;

procedure TCnMsgChild.FormDestroy(Sender: TObject);
begin
  CnLanguageManager.RemoveChangeNotifier(LanguageChanged);
  CnViewerOptions.MsgColumnWidth := MsgTree.Header.Columns[1].Width;

  FFilter.Free;
  FViewStore.Free;
  if FStore <> nil then
  begin
    CnMsgManager.RemoveStore(FStore);
    FStore := nil;
  end;
end;

procedure TCnMsgChild.OnStoreChange(Sender: TObject;
  Operation: TCnStoreChangeType; StartIndex, EndIndex: Integer);
begin
  if Sender is TCnMsgStore then
  begin
    case Operation of
      ctProcess:
        begin
          ProcessID := (Sender as TCnMsgStore).ProcessID;
          ProcName := (Sender as TCnMsgStore).ProcName;
          CnMainViewer.UpdateFormInSwitch(Self, fsUpdate);
        end;
      ctAdd:
        begin
          AddBatchItemToView(Sender as TCnMsgStore, StartIndex, EndIndex);
        end;
      ctModify:
        begin

        end;
      ctTimeChanged:
        begin
          RefreshTime(Sender);
        end;
    else
      ; // Other situation
    end;
  end;
end;

procedure TCnMsgChild.SetStore(const Value: TCnMsgStore);
begin
  if FStore <> Value then
  begin
    if FStore <> nil then
      CnMsgManager.RemoveStore(FStore);
    FStore := Value;
    if FStore <> nil then
    begin
      FStore.OnChange := OnStoreChange;
      FProcessID := FStore.ProcessID;
      FProcName := FStore.ProcName;
    end;
  end;
end;

procedure TCnMsgChild.FormActivate(Sender: TObject);
begin
  if not CnMainViewer.UpdatingSwitch and not CnMainViewer.ClickingSwitch then
    CnMainViewer.UpdateFormInSwitch(Self, fsActiveChange);
end;

procedure TCnMsgChild.TreeGetText(Sender: TBaseVirtualTree;
  Node: PVirtualNode; Column: TColumnIndex; TextType: TVSTTextType;
  var CellText: WideString);
var
  Index: Integer;
  MsgItem: TCnMsgItem;
begin
  if FViewStore = nil then Exit;
  
  Index := Node^.AbsoluteIndex - 1; //.AbsoluteIndex(Node);
  // ԭʼ��VirtualTree�У���ø� Index ����Ӱ����ʾ�ٶȣ�
  // ���޸�VirtualTreeԴ��������ֻ֧��˳�����ӵĽڵ�

  if Index >= FViewStore.MsgCount then Exit;

  MsgItem := FViewStore.Msgs[Index];
  if MsgItem = nil then Exit;

  case Column of
    0: CellText := IntToStr(Index + 1);                                  // ���
    1: CellText := MsgItem.Msg;                           // ����
    2: CellText := SCnMsgTypeDescArray[MsgItem.MsgType]^; // ����
    3: CellText := IntToStr(MsgItem.Level);               // ���
    4: CellText := '$' + IntToHex(MsgItem.ThreadId, 2);   // �߳� ID
    5: CellText := MsgItem.Tag;
    6: CellText := GetTimeDesc(MsgItem);
  else
    CellText := '';
  end;
end;

procedure TCnMsgChild.UpdateToViewStore(Source, Dest: TCnMsgStore);
var
  I: Integer;
begin
  if (Source = nil) or (Dest = nil) then Exit;
  Dest.ClearMsgs;
  for I := 0 to Source.MsgCount - 1 do
    if FFilter.CheckVisible(Source.Msgs[I]) then
      Dest.AddAMsgItem(Source.Msgs[I]);
end;

procedure TCnMsgChild.InitTree;
const
  CnColumnsWidth: array[0..6] of Integer = (36, 230, 55, 41, 77, 60, 84);
var
  I: Integer;
begin
  if FMsgTree = nil then
    Exit;

  FMsgTree.Align := alClient;
  FMsgTree.DefaultNodeHeight := 16;
  FMsgTree.LineStyle := lsSolid;
  FMsgTree.Hint := SCnHintMsgTree;
  FMsgTree.Header.Options := FMsgTree.Header.Options + [hoVisible];
  FMsgtree.TreeOptions.SelectionOptions := FMsgtree.TreeOptions.SelectionOptions
    + [toFullRowSelect, toMiddleClickSelect, toRightClickSelect, toMultiSelect];
  FMsgTree.TreeOptions.PaintOptions := FMsgTree.TreeOptions.PaintOptions + [toHideFocusRect];
  FMsgTree.TreeOptions.AutoOptions := FMsgTree.TreeOptions.AutoOptions + [toAutoExpand, toAutoScroll];
  FMsgTree.OnGetText := TreeGetText;
  FMsgTree.OnChange := TreeChange;
  FMsgTree.OnEnter := TreeEnter;
  FMsgTree.OnKeyPress := TreeKeyPress;
  FMsgTree.OnColumnResize := TreeColumnResize;
  FMsgTree.OnBeforeItemPaint := TreeBeforeItemPaint;
  FMsgTree.OnBeforeItemErase := TreeBeforeItemErase;

  FMsgTree.Parent := pnlTree;
  FMsgTree.PopupMenu := pmTree;
  for I := Low(SCnTreeColumnArray) to High(SCnTreeColumnArray) do
  begin
    with FMsgTree.Header.Columns.Add do
    begin
      Text := SCnTreeColumnArray[I]^;
      if (I <> 1) or (CnViewerOptions.MsgColumnWidth < CnColumnsWidth[I]) then
        Width := CnColumnsWidth[I]
      else
        Width := CnViewerOptions.MsgColumnWidth; // ��Ϣ�п�
    end;
  end;

  FMsgTree.Header.MainColumn := 1; // "�����Ϣ" ��
  FMsgTree.Header.Columns[0].MinWidth := CnColumnsWidth[0];
  FMsgTree.Header.Options := FMsgTree.Header.Options - [hoDrag];
  for I := 0 to FMsgTree.Header.Columns.Count - 1 do
    FMsgTree.OnColumnResize(FMsgTree.Header, I);

  DoTreeResize;
end;

procedure TCnMsgChild.InitTimeTree;
const
  CnColumnsWidth: array[0..6] of Integer = (20, 37, 64, 64, 64, 64, 40);
var
  I: Integer;
begin
  if FTimeTree = nil then
    Exit;

  FTimeTree.Align := alClient;
  FTimeTree.DefaultNodeHeight := 16;
  FTimeTree.LineStyle := lsDotted;
  FTimeTree.Hint := SCnTimeHint;
  FTimeTree.Header.Options := FTimeTree.Header.Options + [hoVisible];
  FTimeTree.TreeOptions.SelectionOptions := FTimeTree.TreeOptions.SelectionOptions
    + [toFullRowSelect, toMiddleClickSelect, toRightClickSelect, toMultiSelect];
  FTimeTree.TreeOptions.PaintOptions := FTimeTree.TreeOptions.PaintOptions + [toHideFocusRect];
  FTimeTree.TreeOptions.AutoOptions := FTimeTree.TreeOptions.AutoOptions + [toAutoExpand, toAutoScroll];

  FTimeTree.OnGetText := TimeGetText;
  FTimeTree.OnChange := TimeChange;
  FTimeTree.OnEnter := TimeEnter;

  FTimeTree.Parent := pnlTime;
  for I := Low(SCnTimeColumnArray) to High(SCnTimeColumnArray) do
  begin
    with FTimeTree.Header.Columns.Add do
    begin
      Text := SCnTimeColumnArray[I]^;
      Width := CnColumnsWidth[I];
    end;
  end;

  FTimeTree.Header.Options := FTimeTree.Header.Options - [hoDrag];
end;

procedure TCnMsgChild.HeadPanelResize(Sender: TObject);
var
  I: Integer;
begin
  if Sender is TPanel then
    for I := 0 to (Sender as TPanel).ControlCount - 1 do
      if (Sender as TPanel).Controls[I] is TComboBox then
        ((Sender as TPanel).Controls[I] as TComboBox).Width := (Sender as TPanel).Width - 1;
end;

procedure TCnMsgChild.TreeColumnResize(Sender: TVTHeader;
  Column: TColumnIndex);
var
  APanel: TPanel;
begin
  case Column of
    0: APanel := pnlLabel;
    1: APanel := pnlMsg;
    2: APanel := pnlType;
    3: APanel := pnlLevel;
    4: APanel := pnlThread;
    5: APanel := pnlTag;
    6: APanel := nil;
  else
    APanel := nil;
  end;
  if APanel = nil then Exit;
  APanel.Width := FMsgTree.Header.Columns[Column].Width + Column mod 2;
  // С���ɣ����� Width ������ۼ�
end;

procedure TCnMsgChild.InitControls;
var
  I: Integer;
begin
  cbbLevel.Items.Clear;
  for I := CnDefLevel downto 0 do
    cbbLevel.Items.Add('<=' + IntToStr(I));
  cbbLevel.ItemIndex := 0;

  cbbType.Items.Clear;
  for I := Ord(Low(CnMsgTypesArray)) to Ord(High(CnMsgTypesArray)) do
    cbbType.Items.Add(SCnMsgTypesDescArray[I]^);
  cbbType.ItemIndex := 0;

  cbbThread.Items.Clear;
  cbbThread.Items.Add('*');
  cbbThread.ItemIndex := 0;
  pnlTree.OnResize := pnlTreeOnResize;

  InitTree;
  InitTimeTree;

  for I := Low(FBookmarks) to High(FBookmarks) do
    FBookmarks[I] := CnInvalidLine;
  btnBookmark.Enabled := FHasBookmarks;

  InitFont;
end;

procedure TCnMsgChild.InitFont;
begin
  if CnViewerOptions.DisplayFont <> nil then
  begin
    FMsgTree.Font.Assign(CnViewerOptions.DisplayFont);
    FTimeTree.Font.Assign(CnViewerOptions.DisplayFont);
    mmoDetail.Font := CnViewerOptions.DisplayFont;
  end;
end;

procedure TCnMsgChild.UpdateConditionsToView(Content: TCnFilterUpdates);
var
  I, OldItemIndex: Integer;
  OldOnChange: TNotifyEvent;
begin
  if fuThreadId in Content then  // �����µ� ThreadID ����
  begin
    OldOnChange := cbbThread.OnChange;
    try
      OldItemIndex := cbbThread.ItemIndex;

      cbbThread.OnChange := nil;
      cbbThread.Items.Clear;
      cbbThread.Items.Add('*');
      for I := 0 to FFilter.Conditions.ThreadIDs.Count - 1 do
        if FFilter.Conditions.ThreadIDs[I] <> nil then
          cbbThread.Items.Add('$' + IntToHex(Integer(FFilter.Conditions.ThreadIDs[I]), 2));

      if cbbThread.Items.Count > 0 then
        cbbThread.ItemIndex := OldItemIndex;
    finally
      cbbThread.OnChange := OldOnChange;
    end;
  end;

  if fuTag in Content then // �����µ� Tags ����
  begin
    OldOnChange := cbbTag.OnChange;
    try
      OldItemIndex := cbbTag.ItemIndex;
      if OldItemIndex = -1 then
        OldItemIndex := 0;

      cbbTag.OnChange := nil;
      cbbTag.Items.Clear;
      for I := 0 to FFilter.Conditions.Tags.Count - 1 do
        cbbTag.Items.Add(FFilter.Conditions.Tags[I]);

      if cbbTag.Items.Count > 0 then
        cbbTag.ItemIndex := OldItemIndex;
    finally
      cbbTag.OnChange := OldOnChange;
    end;
  end;
end;

procedure TCnMsgChild.FilterChange(Sender: TObject);
begin
  // ���ݲ������޸� Filter��Ȼ����½���
  if Sender = cbbType then
  begin
    FFilter.MsgTypes := CnMsgTypesArray[cbbType.ItemIndex];
  end
  else if Sender = cbbThread then
  begin
    if cbbThread.ItemIndex <= 0 then
      FFilter.ThreadId := 0
    else
      FFilter.ThreadId := Cardinal(FFilter.Conditions.ThreadIDs[cbbThread.ItemIndex])
    // ThreadIDs �ж��˵�һ�� 0 ͨ�䣬�� cbbThread�е�һ��Ҳ�� *��������ȣ������һ
  end

  else if Sender = cbbLevel then
    FFilter.Level := CnDefLevel - cbbLevel.ItemIndex
  else if Sender = cbbTag then
    FFilter.Tag := cbbTag.Text;

  UpdateToViewStore(FStore, FViewStore);
  UpdateViewStoreToTree;
  mmoDetail.Clear;
end;

procedure TCnMsgChild.UpdateViewStoreToTree;
var
  I, OldIndent: Integer;
  Node: PVirtualNode;
begin
  FMsgTree.Clear;
  if FViewStore.MsgCount = 0 then
    Exit;

  OldIndent := FViewStore.Msgs[0].Indent;

  for I := 0 to FViewStore.MsgCount - 1 do
  begin
    Node := FMsgTree.GetLast;
    AddAItemToTree(OldIndent, Node, FViewStore.Msgs[I]);
  end;
end;

procedure TCnMsgChild.AddAItemToTree(var OldIndent: Integer;
  var PrevNode: PVirtualNode; AItem: TCnMsgItem);
var
  Indent: Integer;
  Parent, OldParent: PVirtualNode;
begin
  Indent := AItem.Indent;
  if Indent = OldIndent then      // ͬ�㣬��ǰһ�ڵ�ĸ��ڵ������ӽڵ�
  begin
    if PrevNode = nil then
      PrevNode := FMsgTree.AddChild(nil, nil)  // ǰһ�ڵ��޸��ڵ㣬�������֮
    else
      PrevNode := FMsgTree.AddChild(PrevNode.Parent, nil);
  end
  else if Indent > OldIndent then // ��һ�㣬��ǰһ�ڵ���ӽڵ�
  begin
    OldParent := PrevNode;
    PrevNode := FMsgTree.AddChild(PrevNode, nil);
    FMsgTree.FullExpand(OldParent);
  end
  else // ��һ�㣬��ǰһ�ڵ�ĸ��ڵ�ĸ��ڵ������ӽڵ�
  begin
    if PrevNode = nil then
      Parent := nil
    else
      Parent := PrevNode.Parent;

    // ������ڵ��Ǹ��ڵ㣬������ȡ Parent �ˣ�����ͻ����
    if (Parent <> nil) and (Parent <> FMsgTree.RootNode) then
      Parent := Parent.Parent;

    PrevNode := FMsgTree.AddChild(Parent, nil);
  end;

  OldIndent := Indent;
end;

procedure TCnMsgChild.AddATimeToTree(AItem: TCnTimeItem);
begin
  FTimeTree.AddChild(nil, nil);
end;

procedure TCnMsgChild.UpdateTimeToTree(AStore: TCnMsgStore);
var
  I: Integer;
  Node: PVirtualNode;
begin
  FTimeTree.Clear;
  if AStore.TimeCount = 0 then
    Exit;

  for I := 0 to AStore.TimeCount - 1 do
    AddATimeToTree(AStore.Times[I]);
end;

procedure TCnMsgChild.TreeChange(Sender: TBaseVirtualTree; Node: PVirtualNode);
var
  Index: Integer;
begin
  if (FViewStore <> nil) and (Node <> nil) then
  begin
    Index := Node^.AbsoluteIndex - 1;
    FSelectedIndex := Index;
    mmoDetail.Text := DescriptionOfMsg(Index, FViewStore.Msgs[Index]);
    FMemContent := mcMsg;
  end;
end;

procedure TCnMsgChild.TreeBeforeItemPaint(Sender: TBaseVirtualTree;
  TargetCanvas: TCanvas; Node: PVirtualNode; ItemRect: TRect;
  var CustomDraw: Boolean);
const
  SCnMsgCustomDrawTypes: TCnMsgTypes = [cmtSeparator];
var
  Index, WidthStart: Integer;
  AMsgItem: TCnMsgItem;
  AText: string;
  ATree: TVirtualStringTree;
begin
  if FViewStore = nil then Exit;
  
  Index := Node^.AbsoluteIndex - 1;
  // ԭʼ�� VirtualTree ��ø� Index ����Ӱ����ʾ�ٶȣ�Ŀǰ�Ż����޴������ˡ�
  AMsgItem := FViewStore.Msgs[Index];
  if AMsgItem = nil then Exit;
      
  CustomDraw := (AMsgItem.MsgType in SCnMsgCustomDrawTypes);
  if not CustomDraw then Exit;

  ATree := Sender as TVirtualStringTree;
  case AMsgItem.MsgType of
    cmtSeparator:
      begin
        with TargetCanvas do
        begin
          Font := ATree.Font;
          if vsSelected in Node.States then
          begin
            Brush.Color := ATree.Colors.SelectionRectangleBlendColor;
            Font.Color := clHighlightText;
          end
          else if AMsgItem.Bookmarked then
            TargetCanvas.Brush.Color := $0066CC66
          else
            Brush.Color := ATree.Color;

          FillRect(ItemRect);
          Pen.Color := clRed;
          Pen.Width := 3;

          AText := IntToStr(Index + 1);
          if TextWidth(AText) + ATree.Margin + ATree.TextMargin < ATree.Header.Columns[0].Width then
            TextOut(ATree.TextMargin + ATree.Margin, ATree.Margin div 2, AText);
            
          WidthStart := ATree.Header.Columns[0].Width + 1;
          MoveTo(WidthStart, (ItemRect.Top + ItemRect.Bottom) div 2);
          LineTo(ItemRect.Right, (ItemRect.Top + ItemRect.Bottom) div 2);
        end;
      end;
  else
    ;
  end;
end;


procedure TCnMsgChild.TimeChange(Sender: TBaseVirtualTree;
  Node: PVirtualNode);
var
  Index: Integer;
begin
  if (FViewStore <> nil) and (Node <> nil) then
  begin
    mmoDetail.Clear;
    mmoDetail.Text := DescriptionOfTime(Node^.AbsoluteIndex - 1);
    FMemContent := mcTime;
  end;
end;

procedure TCnMsgChild.TimeGetText(Sender: TBaseVirtualTree;
  Node: PVirtualNode; Column: TColumnIndex; TextType: TVSTTextType;
  var CellText: WideString);
var
  Index: Integer;
  ATimeItem: TCnTimeItem;
begin
  Index := Node^.AbsoluteIndex - 1;
  if (FStore = nil) or (Index > FStore.TimeCount) then
    Exit;

  ATimeItem := FStore.Times[Index];
  if ATimeItem = nil then
    Exit;

  case Column of
    0: CellText := IntToStr(Index + 1);                                  // ���
    1: CellText := IntToStr(ATimeItem.PassCount);                        // ����
    2: CellText := Format('%f', [ATimeItem.CPUPeriod / CPUClock]);       // ��ʱ��
    3: CellText := Format('%f', [ATimeItem.AvePeriod / CPUClock]);       // ƽ��ʱ��
    4: CellText := Format('%f', [ATimeItem.MaxPeriod / CPUClock]);       // ���ʱ��
    5: CellText := Format('%f', [ATimeItem.MinPeriod / CPUClock]);       // ��Сʱ��
    6: CellText := ATimeItem.Tag;
  else
    CellText := '';
  end;
end;

procedure TCnMsgChild.ClearStores;
begin
  FMsgTree.Clear;
  FViewStore.ClearMsgs;
  if FStore <> nil then
    FStore.ClearMsgs;
  if FMemContent = mcMsg then
    mmoDetail.Clear;
end;

procedure TCnMsgChild.cbbSearchKeyPress(Sender: TObject; var Key: Char);
begin
  if Key = #13 then
  begin
    if cbbSearch.Text <> '' then
    begin
      btnSearch.Click;
      if (cbbSearch.Items.IndexOf(cbbSearch.Text) < 0) then
      begin
        if (cbbSearch.Items.Count >= CnViewerOptions.SearchDownCount) then
          cbbSearch.Items.Delete(cbbSearch.Items.Count - 1);
        cbbSearch.Items.Insert(0, cbbSearch.Text);
      end;
    end;
    Key := #0;
  end;
end;

function TCnMsgChild.DescriptionOfMsg(Index: Integer; AMsgItem: TCnMsgItem): string;
begin
  if AMsgItem <> nil then
    Result := Format(SCnMsgDescriptionFmt, [Index + 1, AMsgItem.Indent,
      AMsgItem.Level, AMsgItem.ThreadId, AMsgItem.ProcessId, AMsgItem.Tag,
      {AMsgItem.MsgCPInterval, } GetLongTimeDesc(AMsgItem), AMsgItem.Msg])
  else
    Result := '';
end;

function TCnMsgChild.DescriptionOfTime(Index: Integer): string;
var
  dTime, aTime, MaxTime, MinTime: Double;
begin
  Result := '';
  if FStore = nil then
    Exit;

  // Add Sesame 2008-1-22 ����ʱ���뷽ʽ��ʾʱ��
  dTime := FStore.Times[Index].CPUPeriod / CPUClock;
  aTime := FStore.Times[Index].AvePeriod / CPUClock;
  MaxTime := FStore.Times[Index].MaxPeriod / CPUClock;
  MinTime := FStore.Times[Index].MinPeriod / CPUClock;

  if (Index >= 0) and (Index < FStore.TimeCount) then
    Result := Format(SCnTimeDescriptionFmt, [Index + 1, FStore.Times[Index].PassCount,
      //FStore.Times[Index].Tag, FStore.Times[Index].CPUPeriod / CPUClock]);
      FStore.Times[Index].Tag, dTime, FStore.UsToTime(dTime),
      aTime, FStore.UsToTime(aTime), MaxTime, FStore.UsToTime(MaxTime),
      MinTime, FStore.UsToTime(MinTime)]);
end;

procedure TCnMsgChild.ClearTimes;
begin
  if FStore <> nil then
    FStore.ClearTimes;

  FTimeTree.Clear;
  if FMemContent = mcTime then
    mmoDetail.Clear;
end;

procedure TCnMsgChild.TreeEnter(Sender: TObject);
var
  Tree: TVirtualStringTree;
begin
  if Sender is TVirtualStringTree then
  begin
    Tree := Sender as TVirtualStringTree;
    if Assigned(Tree.OnChange) then
      Tree.OnChange(Tree, Tree.FocusedNode);
  end;
end;

procedure TCnMsgChild.TimeEnter(Sender: TObject);
var
  Tree: TVirtualStringTree;
begin
  if Sender is TVirtualStringTree then
  begin
    Tree := Sender as TVirtualStringTree;
    if Assigned(Tree.OnChange) then
      Tree.OnChange(Tree, Tree.FocusedNode);
  end;
end;

procedure TCnMsgChild.LoadFromFile(const FileName: string);
var
  Filer: ICnMsgFiler;
  Ext: string;
  DumpFile: TFileStream;
  AMsgDesc: TCnMsgDesc;
  MsgSize: Integer;
begin
  if (FStore <> nil) and FileExists(FileName) then
  begin
    Screen.Cursor := crHourGlass;
    try
      ClearStores;
      ClearTimes;
      ClearBookMarks;

      Ext := LowerCase(_CnExtractFileExt(FileName));
      if Ext = '.xml' then // �� CnDebugViewer ����� XML �ļ�
      begin
        Filer := TCnMsgXMLFiler.Create;
        FStore.LoadFromFile(Filer, FileName);
        AddBatchItemToView(FStore, 0, FStore.MsgCount - 1);
        RefreshTime(FStore);
      end
      else if Ext = '.json' then // �� CnDebugViewer ����� JSON �ļ�
      begin
        Filer := TCnMsgJSONFiler.Create;
        FStore.LoadFromFile(Filer, FileName);
        AddBatchItemToView(FStore, 0, FStore.MsgCount - 1);
        RefreshTime(FStore);
      end
      else if Ext = '.cdd' then // �� CnDebug.pas ֱ�� Dump �����ļ�
      begin
        DumpFile := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
        try
          FStore.BeginUpdate;
          try
            while DumpFile.Position < DumpFile.Size do
            begin
              DumpFile.Read(MsgSize, SizeOf(MsgSize));
              if MsgSize > SizeOf(TCnMsgDesc) then
                Break; // ��������
              DumpFile.Seek(0 - SizeOf(MsgSize), soFromCurrent); // ����һ��
              DumpFile.Read(AMsgDesc, MsgSize);
              FStore.AddMsgDesc(@AMsgDesc);
            end;
          except
            ; // ������
          end;
        finally
          FStore.EndUpdate;
        end;
      end;
    finally
      Screen.Cursor := crDefault;
    end;
  end;
end;

procedure TCnMsgChild.SaveToFile(const FileName: string);
var
  Filer: ICnMsgFiler;
begin
  if FStore <> nil then
  begin
    Screen.Cursor := crHourGlass;
    try
      if LowerCase(_CnExtractFileExt(FileName)) = '.xml' then
        Filer := TCnMsgXMLFiler.Create
      else
        Filer := TCnMsgJsonFiler.Create;
      FStore.SaveToFile(Filer, FileName);
    finally
      Screen.Cursor := crDefault;
    end;
  end;
end;

procedure TCnMsgChild.AddBatchItemToView(AStore: TCnMsgStore; StartIndex,
  EndIndex: Integer);
var
  I, Slot, OldIndent: Integer;
  UpdateContent: TCnFilterUpdates;
  AMsgItem: TCnMsgItem;
  PrevNode, LastNode: PVirtualNode;
begin
  if (AStore = nil) or (StartIndex < 0) or (EndIndex < 0) then Exit;
  for I := StartIndex to EndIndex do
  begin
    AMsgItem := AStore.Msgs[I];
    UpdateContent := [];
    if FFilter.Conditions.CheckAndAddThreadID(AMsgItem.ThreadId) then
      Include(UpdateContent, fuThreadId);
    if FFilter.Conditions.CheckAndAddTag(AMsgItem.Tag) then
      Include(UpdateContent, fuTag);
    UpdateConditionsToView(UpdateContent);

    // if not FFilter.Filtered or not FFilter.CheckVisible(AMsgItem) then
    if FFilter.Filtered and not FFilter.CheckVisible(AMsgItem) then
    begin
      // �������ɼ��򲻴���
    end
    else
    begin
      // FStore ���ɶ�д�̸߳��£��˴����ƿɼ� Item �� ViewStore �С�
      if FViewStore.MsgCount > 0 then
        OldIndent := FViewStore.Msgs[FViewStore.MsgCount - 1].Indent
      else
        OldIndent := 0;
      FViewStore.AddAMsgItem(AMsgItem);
      PrevNode := FMsgTree.GetLast;
      AddAItemToTree(OldIndent, PrevNode, AMsgItem);

      if AMsgItem.Bookmarked then
      begin
        FHasBookmarks := True;
        Slot := GetAnEmptyBookmarkSlot();
        if Slot <> CnInvalidSlot then
        begin
          LastNode := FMsgTree.GetLast;
          if LastNode <> nil then
          begin
            SetSlotToBookmark(Slot, LastNode.AbsoluteIndex - 1);
            // Line ���� Index���� 0 ��ʼ
            UpdateBookmarkMenu;
          end;
        end;
      end;

      // FMsgTree �������¹�
      if CnViewerOptions.AutoScroll then
        PostMessage(FMsgTree.Handle, WM_TREE_GOTOLAST, 0, 0);
    end;
  end;
end;

procedure TCnMsgChild.RefreshTime(Sender: TObject);
begin
  // ˢ�� TimeTree
  UpdateTimeToTree(Sender as TCnMsgStore);
  FTimeTree.Refresh;
end;

procedure TCnMsgChild.FindNode(const AText: string; IsDown: Boolean;
  IsSeperator: Boolean);
var
  I, OldPos: Integer;
  FoundNode: PVirtualNode;
begin
  if (AText = '') or (FStore = nil) then
    Exit;

  if FMsgTree.FocusedNode <> nil then
    OldPos := FMsgTree.FocusedNode.AbsoluteIndex // �� 1 ��ʼ��
  else
    OldPos := 0;

  try
    Screen.Cursor := crHourGlass;
    if IsDown then
    begin
      for I := OldPos to FViewStore.MsgCount - 1 do // ��ѡ�е���һ����ʼ����
      begin
        if CheckFind(FViewStore.Msgs[I], AText, IsSeperator) then
        begin
          FoundNode := FMsgTree.GetNodeByAbsoluteIndex(I + 1);
          FMsgTree.FocusedNode := FoundNode;
          FMsgTree.Selected[FoundNode] := True;
          Exit;
        end;
      end;

      if OldPos > 0 then
      begin
        for I := 0 to OldPos - 1 do // ѭ������
        begin
          if CheckFind(FViewStore.Msgs[I], AText, IsSeperator) then
          begin
            FoundNode := FMsgTree.GetNodeByAbsoluteIndex(I + 1);
            FMsgTree.FocusedNode := FoundNode;
            FMsgTree.Selected[FoundNode] := True;
            Exit;
          end;
        end;
      end;
    end
    else // ������
    begin
      if OldPos > 0 then
      begin
        for I := OldPos - 1 downto 0 do // ѭ������
        begin
          if CheckFind(FViewStore.Msgs[I], AText, IsSeperator) then
          begin
            FoundNode := FMsgTree.GetNodeByAbsoluteIndex(I + 1);
            FMsgTree.FocusedNode := FoundNode;
            FMsgTree.Selected[FoundNode] := True;
            Exit;
          end;
        end;
      end;

      for I := FViewStore.MsgCount - 1 downto OldPos do // ��ѡ�е���һ����ʼ����
      begin
        if CheckFind(FViewStore.Msgs[I], AText, IsSeperator) then
        begin
          FoundNode := FMsgTree.GetNodeByAbsoluteIndex(I + 1);
          FMsgTree.FocusedNode := FoundNode;
          FMsgTree.Selected[FoundNode] := True;
          Exit;
        end;
      end;
    end;
  finally
    Screen.Cursor := crDefault;
  end;
  // ��ʾδ�ҵ�
  ErrorDlg(SCnNotFound);
end;

function TCnMsgChild.CheckFind(AItem: TCnMsgItem;
  const AText: string; IsSeperator: Boolean): Boolean;
begin
  Result := False;
  if (AText = '') or (AItem = nil) then Exit;
  if IsSeperator then
    Result := AItem.MsgType = cmtSeparator
  else
    Result := Pos(UpperCase(AText), UpperCase(AItem.Msg)) > 0;
end;

procedure TCnMsgChild.btnSearchClick(Sender: TObject);
begin
  if Trim(cbbSearch.Text) <> '' then
    FindNode(cbbSearch.Text, True);
end;

procedure TCnMsgChild.TreeKeyPress(Sender: TObject; var Key: Char);
begin
  if Key in ['0'..'9', 'A'..'z'] then
  begin
    cbbSearch.SetFocus;
    PostMessage(cbbSearch.Handle, WM_CHAR, Integer(Key), 0);
  end;
end;

procedure TCnMsgChild.DoCreate;
begin
  inherited;
  CnLanguageManager.AddChangeNotifier(LanguageChanged);
  CnLanguageManager.TranslateForm(Self);
end;

procedure TCnMsgChild.LanguageChanged(Sender: TObject);
var
  I, OldIndex: Integer;
begin
  FMsgTree.Hint := SCnHintMsgTree;
  for I := Low(SCnTreeColumnArray) to High(SCnTreeColumnArray) do
    FMsgTree.Header.Columns[I].Text := SCnTreeColumnArray[I]^;

  FTimeTree.Hint := SCnTimeHint;
  for I := Low(SCnTimeColumnArray) to High(SCnTimeColumnArray) do
    FTimeTree.Header.Columns[I].Text := SCnTimeColumnArray[I]^;

  OldIndex := cbbType.ItemIndex;
  cbbType.Items.Clear;
  cbbType.Items.Add(SCnMsgTypeNone);
  for I := Ord(Low(TCnMsgType)) to Ord(High(TCnMsgType)) do
    cbbType.Items.Add(SCnMsgTypeDescArray[TCnMsgType(I)]^);
  cbbType.ItemIndex := OldIndex;

  if FProcName = '' then
    CnMainViewer.UpdateFormInSwitch(Self, fsUpdate);
end;

function TCnMsgChild.GetAnEmptyBookmarkSlot: Integer;
var
  I: Integer;
begin
  Result := CnInvalidSlot;
  for I := Low(FBookmarks) to High(FBookmarks) do
  begin
    if FBookmarks[I] = CnInvalidLine then
    begin
      Result := I;
      Exit;
    end;
  end;
end;

procedure TCnMsgChild.ReleaseAnBookmarkSlot(Slot: Integer);
begin
  if (FStore <> nil) and (Slot >= Low(FBookmarks)) and (Slot <= High(FBookmarks))
    and (FBookmarks[Slot] <> CnInvalidSlot) then
  begin
    if (FBookmarks[Slot] >= 0) and (FBookmarks[Slot] < FViewStore.MsgCount) then
      FViewStore.Msgs[FBookmarks[Slot]].Bookmarked := False;
    FBookmarks[Slot] := CnInvalidLine;
    FMsgTree.Refresh;
  end;
end;

function TCnMsgChild.GetSlotFromBookmarkLine(Line: Integer): Integer;
var
  I: Integer;
begin
  Result := CnInvalidSlot;
  for I := Low(FBookmarks) to High(FBookmarks) do
  begin
    if FBookmarks[I] = Line then
    begin
      Result := I;
      Exit;
    end;
  end;
end;

procedure TCnMsgChild.SetSlotToBookmark(Slot, Line: Integer);
begin
  if (FStore <> nil) and (Slot >= Low(FBookmarks)) and (Slot <= High(FBookmarks)) then
  begin
    FViewStore.Msgs[Line].Bookmarked := True;
    FBookmarks[Slot] := Line;
    FMsgTree.Refresh;
  end;
end;

procedure TCnMsgChild.ToggleBookmark;
var
  Slot: Integer;
begin
  if FMsgTree.ObtainFirstSelection <> nil then
    FSelectedIndex := FMsgTree.ObtainFirstSelection^.AbsoluteIndex - 1;

  Slot := GetSlotFromBookmarkLine(FSelectedIndex);
  if Slot = CnInvalidSlot then
  begin
    Slot := GetAnEmptyBookmarkSlot;

    if Slot <> CnInvalidSlot then
      SetSlotToBookmark(Slot, FSelectedIndex)
    else
      ErrorDlg(SCnBookmarkFull);
  end
  else
    ReleaseAnBookmarkSlot(Slot);
  UpdateBookmarkMenu;
end;

procedure TCnMsgChild.TreeBeforeItemErase(Sender: TBaseVirtualTree;
  TargetCanvas: TCanvas; Node: PVirtualNode; ItemRect: TRect;
  var ItemColor: TColor; var EraseAction: TItemEraseAction);
var
  Index: Integer;
  AMsgItem: TCnMsgItem;
begin
  if FStore = nil then Exit;
  Index := Node^.AbsoluteIndex - 1;

  AMsgItem := FViewStore.Msgs[Index];
  if AMsgItem = nil then Exit;

  EraseAction := eaDefault;
  if AMsgItem.Bookmarked then
  begin
    ItemColor := $0066CC66;
    EraseAction := eaColor;
  end;
end;

procedure TCnMsgChild.UpdateBookmarkMenu;
var
  Item: TMenuItem;
  I, J: Integer;
begin
  J := 0;
  MenuDropBookmark.Clear;
  FHasBookmarks := False;
  for I := Low(FBookmarks) to High(FBookmarks) do
  begin
    if FBookmarks[I] <> CnInvalidLine then
    begin
      Inc(J);
      FHasBookmarks := True;

      Item := TMenuItem.Create(pmTree);
      Item.Caption := Format(SCnBookmark, [J, FBookmarks[I] + 1]);
      Item.Tag := I; // Slot
      Item.ImageIndex := 82;
      Item.OnClick := BookmarkMenuClick;
      MenuDropBookmark.Insert(MenuDropBookmark.Count, Item);
    end;
  end;
  btnBookmark.Enabled := FHasBookmarks;
  UpdateBookmarkToMainMenu;
end;

procedure TCnMsgChild.BookmarkMenuClick(Sender: TObject);
var
  Index: Integer;
  BookmarkVisible: Boolean;
begin
  if (Sender <> nil) and (FStore <> nil) then
  begin
    Index := (Sender as TComponent).Tag;
    if (Index >= Low(FBookmarks)) and (Index <= High(FBookmarks)) then
      Index := Abs(FBookmarks[Index]);

    // ���������ʾ��������
    if (FViewStore.MsgCount > 0) and (Index < FViewStore.MsgCount) then
      BookmarkVisible := FViewStore.Msgs[Index].Bookmarked
    else
      BookmarkVisible := False;

    if not BookmarkVisible then
    begin
      ErrorDlg(SCnBookmarkNOTExist);
      Exit;
    end;

    FMsgTree.Selected[FMsgTree.GetNodeByAbsoluteIndex(Index + 1)] := True;
    FMsgTree.FocusedNode := FMsgTree.GetNodeByAbsoluteIndex(Index + 1);
  end;
end;

procedure TCnMsgChild.UpdateBookmarkToMainMenu;
var
  Item: TMenuItem;
  I, J: Integer;
begin
  J := 0;
  CnMainViewer.MenuJump.Clear;
  for I := Low(FBookmarks) to High(FBookmarks) do
  begin
    if FBookmarks[I] <> CnInvalidLine then
    begin
      Inc(J);

      Item := TMenuItem.Create(CnMainViewer.mmMain);
      Item.Caption := Format(SCnBookmark, [J, FBookmarks[I] + 1]);
      Item.Tag := I;
      Item.ImageIndex := 78;
      Item.OnClick := BookmarkMenuClick;
      CnMainViewer.MenuJump.Insert(CnMainViewer.MenuJump.Count, Item);
    end;
  end;
  CnMainViewer.MenuJump.Enabled := FHasBookmarks;
end;

procedure TCnMsgChild.GotoNextBookmark;
var
  I, Index: Integer;
begin
  if (FStore = nil) or (FMsgTree.TotalCount = 0) or (FMsgTree.FocusedNode = nil) then
    Exit;

  Index := FMsgTree.FocusedNode.AbsoluteIndex - 1;
  // AbsoluteIndex �� 1 ��ʼ

  for I := Index + 1 to FViewStore.MsgCount - 1 do
  begin
    if FViewStore.Msgs[I].Bookmarked then
    begin
      FMsgTree.Selected[FMsgTree.GetNodeByAbsoluteIndex(Index + 1)] := False;
      FMsgTree.Selected[FMsgTree.GetNodeByAbsoluteIndex(I + 1)] := True;
      FMsgTree.FocusedNode := FMsgTree.GetNodeByAbsoluteIndex(I + 1);
      Exit;
    end;
  end;
end;

procedure TCnMsgChild.GotoPrevBookmark;
var
  I, Index: Integer;
begin
  if (FStore = nil) or (FMsgTree.TotalCount = 0) or (FMsgTree.FocusedNode = nil) then
    Exit;

  Index := FMsgTree.FocusedNode.AbsoluteIndex - 1;
  // AbsoluteIndex �� 1 ��ʼ
  if Index = 0 then Exit;

  for I := Index - 1 downto 0 do
  begin
    if FViewStore.Msgs[I].Bookmarked then
    begin
      FMsgTree.Selected[FMsgTree.GetNodeByAbsoluteIndex(Index + 1)] := False;
      FMsgTree.Selected[FMsgTree.GetNodeByAbsoluteIndex(I + 1)] := True;
      FMsgTree.FocusedNode := FMsgTree.GetNodeByAbsoluteIndex(I + 1);
      Exit;
    end;
  end;
end;

procedure TCnMsgChild.ClearBookMarks;
var
  I: Integer;
begin
  for I := Low(FBookmarks) to High(FBookmarks) do
    ReleaseAnBookmarkSlot(I);
end;

procedure TCnMsgChild.pmTreePopup(Sender: TObject);
begin
  MenuDropBookmark.Enabled := FHasBookmarks;
  MenuDropBookmark.Caption := CnMainViewer.MenuJump.Caption;
end;

procedure TCnMsgChild.ClearAllBookmarks;
begin
  ClearBookMarks;
  UpdateBookmarkMenu;
  FMsgTree.Invalidate;
end;

procedure TCnMsgChild.pnlTreeOnResize(Sender: TObject);
begin
  if (not Showing) or IsResizing then
    Exit;

  DoTreeResize;
end;

function SelectionCompare(Item1, Item2: Pointer): Integer;
var
  N1, N2: PVirtualNode;
begin
  if (Item1 <> nil) and (Item2 = nil) then
    Result := 1
  else if (Item1 = nil) and (Item2 <> nil) then
    Result := -1
  else if (Item1 = nil) and (Item2 = nil) then
    Result := 0
  else // ����Ϊ nil
  begin
    N1 := PVirtualNode(Item1);
    N2 := PVirtualNode(Item2);
    Result := N1^.AbsoluteIndex - N2^.AbsoluteIndex;
  end;
end;

function TCnMsgChild.GetSelectedContent: string;
var
  I, Index: Integer;
  List: TList;
  Node: PVirtualNode;
begin
  if FMsgTree.SelectedCount = 1 then
  begin
    if mmoDetail.SelLength > 0 then
      Result := mmoDetail.SelText
    else
      Result := mmoDetail.Lines.Text;
  end
  else
  begin
    List := TList.Create;
    try
      FMsgTree.ObtainSelections(List);
      List.Sort(SelectionCompare);

      for I := 0 to List.Count - 1 do
      begin
        Node := PVirtualNode(List[I]);
        Index := Node^.AbsoluteIndex - 1;
        Result := Result + DescriptionOfMsg(Index, FViewStore.Msgs[Index]) + #13#10#13#10;
      end;
    finally
      List.Free;
    end;
  end;
end;

procedure TCnMsgChild.RequireRefreshTime;
begin
  RefreshTime(FStore);
end;

function TCnMsgChild.GetSelectedItem: TCnMsgItem;
begin
  Result := nil;
  if FMsgTree.SelectedCount = 1 then
    if (FSelectedIndex >= 0) and (FSelectedIndex < FViewStore.MsgCount) then
      Result := FViewStore.Msgs[FSelectedIndex];
end;

function TCnMsgChild.GetSelectedText: string;
var
  I, Index: Integer;
  List: TList;
  Node: PVirtualNode;
begin
  List := TList.Create;
  try
    FMsgTree.ObtainSelections(List);
    List.Sort(SelectionCompare);

    for I := 0 to List.Count - 1 do
    begin
      Node := PVirtualNode(List[I]);
      Index := Node^.AbsoluteIndex - 1;
      Result := Result + FViewStore.Msgs[Index].Msg + #13#10;
    end;
  finally
    List.Free;
  end;
end;

procedure TCnMsgChild.DoTreeResize;
var
  I, Width1: Integer;
begin
  with FMsgTree do
  begin
    Visible := False;
    try
      Width1 := Width - Header.Columns[0].Width - 24;
      if not VertScrollBar.Visible then
        Width1 := Width1 - VertScrollBar.Size;
      for I := Header.Columns.Count - 1 downto 2 do
        Width1 := Width1 - Header.Columns[I].Width;

      Header.Columns[1].Width := Width1;
    finally
      Visible := True;
    end;
  end;
end;

procedure TCnMsgChild.FormShow(Sender: TObject);
begin
  if Assigned(pnlTree.OnResize) then
    pnlTree.OnResize(pnlTree);
end;

end.
