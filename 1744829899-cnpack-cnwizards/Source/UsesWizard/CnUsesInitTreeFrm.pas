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

unit CnUsesInitTreeFrm;
{ |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�����������������Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע��
* ����ƽ̨��PWin7 + Delphi 5.01
* ���ݲ��ԣ�PWin7/10 + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����ô����е��ַ���֧�ֱ��ػ�����ʽ
* �޸ļ�¼��2021.08.21 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNUSESTOOLS}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ComCtrls, StdCtrls, ToolWin, ExtCtrls, ActnList, ToolsAPI,
  CnTree, CnCommon, CnWizMultiLang, CnWizConsts, CnWizUtils, CnWizIdeUtils,
  Menus;

type
  TCnUsesInitTreeForm = class(TCnTranslateForm)
    grpFilter: TGroupBox;
    chkProjectPath: TCheckBox;
    chkSystemPath: TCheckBox;
    grpTree: TGroupBox;
    tvTree: TTreeView;
    pnlTop: TPanel;
    lblProject: TLabel;
    cbbProject: TComboBox;
    tlbUses: TToolBar;
    btnGenerateUsesTree: TToolButton;
    grpInfo: TGroupBox;
    actlstUses: TActionList;
    actGenerateUsesTree: TAction;
    actHelp: TAction;
    actExit: TAction;
    btn1: TToolButton;
    btnHelp: TToolButton;
    btnExit: TToolButton;
    lblSourceFile: TLabel;
    lblDcuFile: TLabel;
    lblSearchType: TLabel;
    lblUsesType: TLabel;
    lblSearchTypeText: TLabel;
    lblUsesTypeText: TLabel;
    actExport: TAction;
    actSearch: TAction;
    btnSearch: TToolButton;
    btnExport: TToolButton;
    btn2: TToolButton;
    actOpen: TAction;
    btnOpen: TToolButton;
    actLocateSource: TAction;
    btnLocateSource: TToolButton;
    pmTree: TPopupMenu;
    Open1: TMenuItem;
    OpeninExplorer1: TMenuItem;
    ExportTree1: TMenuItem;
    Search1: TMenuItem;
    N1: TMenuItem;
    N2: TMenuItem;
    dlgSave: TSaveDialog;
    actSearchNext: TAction;
    btnSearchNext: TToolButton;
    btn4: TToolButton;
    dlgFind: TFindDialog;
    mmInit: TMainMenu;
    File1: TMenuItem;
    Edit1: TMenuItem;
    Search2: TMenuItem;
    Help1: TMenuItem;
    Exit1: TMenuItem;
    AnalyseProject1: TMenuItem;
    ExportTree2: TMenuItem;
    Search3: TMenuItem;
    SearchNext1: TMenuItem;
    N3: TMenuItem;
    Open2: TMenuItem;
    OpeninExplorer2: TMenuItem;
    Help2: TMenuItem;
    SearchNext2: TMenuItem;
    mmoSourceFileText: TMemo;
    mmoDcuFileText: TMemo;
    statUses: TStatusBar;
    grpOrder: TGroupBox;
    mmoOrder: TMemo;
    procedure actGenerateUsesTreeExecute(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure chkSystemPathClick(Sender: TObject);
    procedure tvTreeChange(Sender: TObject; Node: TTreeNode);
    procedure actExitExecute(Sender: TObject);
    procedure actHelpExecute(Sender: TObject);
    procedure actOpenExecute(Sender: TObject);
    procedure actlstUsesUpdate(Action: TBasicAction; var Handled: Boolean);
    procedure actExportExecute(Sender: TObject);
    procedure actSearchExecute(Sender: TObject);
    procedure dlgFindClose(Sender: TObject);
    procedure dlgFindFind(Sender: TObject);
    procedure actSearchNextExecute(Sender: TObject);
    procedure actLocateSourceExecute(Sender: TObject);
  private
    FTree: TCnTree;
    FFileNames: TStringList;
    FLibPaths: TStringList;
    FDcuPath: string;
    FProjectList: TInterfaceList;
    FOldSearchStr: string;
    FInitOrders: TStringList;
    procedure InitProjectList;
    procedure GenerateInitOrder;
    procedure OnDepthFirst(Sender: TObject);
    procedure TreeSaveANode(ALeaf: TCnLeaf; ATreeNode: TTreeNode; var Valid: Boolean);
    procedure SearchAUnit(const AFullDcuName, AFullSourceName: string; ProcessedFiles: TStrings;
      UnitLeaf: TCnLeaf; Tree: TCnTree; AProject: IOTAProject);
    {* �ݹ���ã����������Ҷ�Ӧ dcu ��Դ��� Uses �б����뵽���е� UnitLeaf ���ӽڵ���}
    procedure UpdateTreeView;
    procedure UpdateInfo(Leaf: TCnLeaf);
    function SearchText(const Text: string; ToDown, IgnoreCase, WholeWord: Boolean): Boolean;
  protected
    function GetHelpTopic: string; override;
  public

  end;

{$ENDIF CNWIZARDS_CNUSESTOOLS}

implementation

{$IFDEF CNWIZARDS_CNUSESTOOLS}

{$R *.DFM}

uses
  CnWizShareImages, CnDCU32, CnWizOptions;

const
  csDcuExt = '.dcu';
  csExploreCmdLine = 'EXPLORER.EXE /e, /select, "%s"';

  csSearchTypeStrings: array[Low(TCnModuleSearchType)..High(TCnModuleSearchType)] of PString =
    (nil, @SCnUsesInitTreeSearchInProject, @SCnUsesInitTreeSearchInProjectSearch,
    @SCnUsesInitTreeSearchInSystemSearch);

type
  TCnUsesLeaf = class(TCnLeaf)
  private
    FIsImpl: Boolean;
    FDcuName: string;
    FSearchType: TCnModuleSearchType;
    FSourceName: string;
  public
    property SourceName: string read FSourceName write FSourceName;
    {* Դ�ļ�����·����}
    property DcuName: string read FDcuName write FDcuName;
    {* Dcu �ļ�����·����}
    property SearchType: TCnModuleSearchType read FSearchType write FSearchType;
    {* ������������}
    property IsImpl: Boolean read FIsImpl write FIsImpl;
    {* �����Ƿ��� implementation ����}
  end;

function GetDcuName(const ADcuPath, ASourceFileName: string): string;
begin
  if ADcuPath = '' then
    Result := _CnChangeFileExt(ASourceFileName, csDcuExt)
  else
    Result := _CnChangeFileExt(ADcuPath + _CnExtractFileName(ASourceFileName), csDcuExt);
end;

procedure TCnUsesInitTreeForm.actGenerateUsesTreeExecute(Sender: TObject);
var
  Proj, P: IOTAProject;
  I: Integer;
  ProjDcu: string;
begin
  Proj := nil;
  if cbbProject.ItemIndex <= 0 then // ��ǰ����
  begin
    Proj := CnOtaGetCurrentProject;
    if (Proj = nil) or not IsDelphiProject(Proj) then
      Exit;
  end
  else
  begin
    // �ض����ƵĹ���
    for I := 0 to FProjectList.Count - 1 do
    begin
      P := FProjectList[I] as IOTAProject;
      if cbbProject.Items[cbbProject.ItemIndex] = _CnExtractFileName(P.FileName) then
      begin
        Proj := P;
        Break;
      end;
    end;
  end;

  if (Proj = nil) or not IsDelphiProject(Proj) then
    Exit;

  // ���빤��
  if not CompileProject(Proj) then
  begin
    Close;
    ErrorDlg(SCnUsesCleanerCompileFail);
    Exit;
  end;

  FTree.Clear;
  FFileNames.Clear;
  statUses.SimpleText := '';

  FDcuPath := GetProjectDcuPath(Proj);
  GetLibraryPath(FLibPaths, False);

  with FTree.Root as TCnUsesLeaf do
  begin
    SourceName := CnOtaGetProjectSourceFileName(Proj);
    DcuName := ProjDcu;
    SearchType := mstInProject;
    IsImpl := False;
    Text := _CnExtractFileName(SourceName);
    ProjDcu := GetDcuName(FDcuPath, SourceName);
  end;

  Screen.Cursor := crHourGlass;
  try
    SearchAUnit(ProjDcu, (FTree.Root as TCnUsesLeaf).SourceName, FFileNames,
      FTree.Root, FTree, Proj);
  finally
    Screen.Cursor := crDefault;
  end;

  GenerateInitOrder;
  UpdateTreeView;
end;

procedure TCnUsesInitTreeForm.FormCreate(Sender: TObject);
begin
  FFileNames := TStringList.Create;
  FLibPaths := TStringList.Create;
  FTree := TCnTree.Create(TCnUsesLeaf);
  FProjectList := TInterfaceList.Create;
  tlbUses.ShowHint := WizOptions.ShowHint;

  FTree.OnSaveANode := TreeSaveANode;

  InitProjectList;
  WizOptions.ResetToolbarWithLargeIcons(tlbUses);
  IdeScaleToolbarComboFontSize(cbbProject);
  FInitOrders := TStringList.Create;

  if WizOptions.UseLargeIcon then
  begin
    tlbUses.Height := tlbUses.Height + csLargeToolbarHeightDelta;
    pnlTop.Height := pnlTop.Height + csLargeToolbarHeightDelta;
  end;
end;

procedure TCnUsesInitTreeForm.FormDestroy(Sender: TObject);
begin
  FInitOrders.Free;
  FProjectList.Free;
  FTree.Free;
  FLibPaths.Free;
  FFileNames.Free;
end;

procedure TCnUsesInitTreeForm.InitProjectList;
var
  I: Integer;
  Proj: IOTAProject;
{$IFDEF BDS}
  PG: IOTAProjectGroup;
{$ENDIF}
begin
  CnOtaGetProjectList(FProjectList);
  cbbProject.Items.Clear;

  if FProjectList.Count <= 0 then
    Exit;

  for I := 0 to FProjectList.Count - 1 do
  begin
    Proj := IOTAProject(FProjectList[I]);
    if Proj.FileName = '' then
      Continue;

{$IFDEF BDS}
    // BDS ��ProjectGroup Ҳ֧�� Project �ӿڣ������Ҫȥ��
    if Supports(Proj, IOTAProjectGroup, PG) then
      Continue;
{$ENDIF}

    if not IsDelphiProject(Proj) then
      Continue;

    cbbProject.Items.Add(_CnExtractFileName(Proj.FileName));
  end;

  if cbbProject.Items.Count > 0 then
  begin
    cbbProject.Items.Insert(0, SCnProjExtCurrentProject);
    cbbProject.ItemIndex := 0;
  end;
end;

procedure TCnUsesInitTreeForm.SearchAUnit(const AFullDcuName,
  AFullSourceName: string; ProcessedFiles: TStrings; UnitLeaf: TCnLeaf;
  Tree: TCnTree; AProject: IOTAProject);
var
  St: TCnModuleSearchType;
  ASourceFileName, ADcuFileName: string;
  UsesList: TStringList;
  I, J: Integer;
  Leaf: TCnUsesLeaf;
  Info: TCnUnitUsesInfo;
begin
  // ���� DCU ��Դ��õ� intf �� impl �������б��������� UnitLeaf ��ֱ���ӽڵ�
  // �ݹ���ø÷���������ÿ�������б��е����õ�Ԫ��
  if  not FileExists(AFullDcuName) and not FileExists(AFullSourceName)
    and not CnOtaIsFileOpen(AFullSourceName) then // ��û���沢�һ�û��������ҲҪ����
    Exit;

  UsesList := TStringList.Create;
  try
    if FileExists(AFullDcuName) then // �� DCU �ͽ��� DCU
    begin
      statUses.SimpleText := AFullDcuName;
      Info := TCnUnitUsesInfo.Create(AFullDcuName);
      try
        for I := 0 to Info.IntfUsesCount - 1 do
          UsesList.Add(Info.IntfUses[I]);
        for I := 0 to Info.ImplUsesCount - 1 do
          UsesList.AddObject(Info.ImplUses[I], TObject(True));
      finally
        Info.Free;
      end;
    end
    else // �������Դ��
    begin
      statUses.SimpleText := AFullSourceName;
      ParseUnitUsesFromFileName(AFullSourceName, UsesList);
    end;
    Application.ProcessMessages;

    // UsesList ���õ�������������·�������ҵ�Դ�ļ�������� dcu
    for I := 0 to UsesList.Count - 1 do
    begin
      // �ҵ�Դ�ļ�
      ASourceFileName := GetFileNameSearchTypeFromModuleName(UsesList[I], St, AProject);
      if (ASourceFileName = '') or (ProcessedFiles.IndexOf(ASourceFileName) >= 0) then
        Continue;

      // ���ұ����� dcu�������ڹ������Ŀ¼�Ҳ������ϵͳ�� LibraryPath ��
      ADcuFileName := GetDcuName(FDcuPath, ASourceFileName);
      if not FileExists(ADcuFileName) then
      begin
        // ��ϵͳ�Ķ�� LibraryPath ����
        for J := 0 to FLibPaths.Count - 1 do
        begin
          if FileExists(MakePath(FLibPaths[J]) + UsesList[I] + csDcuExt) then
          begin
            ADcuFileName := MakePath(FLibPaths[J]) + UsesList[I] + csDcuExt;
            Break;
          end;
        end;
      end;

      if not FileExists(ADcuFileName) then
        Continue;

      // ASourceFileName ������δ��������½�һ�� Leaf���ҵ�ǰ Leaf ����
      Leaf := Tree.AddChild(UnitLeaf) as TCnUsesLeaf;
      Leaf.Text := _CnExtractFileName(_CnChangeFileExt(ASourceFileName, ''));
      Leaf.SourceName := ASourceFileName;
      Leaf.DcuName := ADcuFileName;
      Leaf.SearchType := St;
      Leaf.IsImpl := UsesList.Objects[I] <> nil;

      ProcessedFiles.Add(ASourceFileName);
      SearchAUnit(ADcuFileName, ASourceFileName, ProcessedFiles, Leaf, Tree, AProject);
    end;
  finally
    UsesList.Free;
  end;
end;

procedure TCnUsesInitTreeForm.OnDepthFirst(Sender: TObject);
var
  Leaf: TCnUsesLeaf;
begin
  Leaf := Sender as TCnUsesLeaf;

  if (Leaf.SearchType = mstInProject) or
    (chkSystemPath.Checked and (Leaf.SearchType = mstSystemSearch)) or
    (chkProjectPath.Checked and (Leaf.SearchType = mstProjectSearch)) then
    FInitOrders.Add(Leaf.Text);
end;

procedure TCnUsesInitTreeForm.GenerateInitOrder;
begin
  FInitOrders.Clear;
  FTree.OnDepthFirstTravelLeaf := OnDepthFirst;

  FTree.DepthFirstTravel(False);

  mmoOrder.Lines.Clear;
  mmoOrder.Lines.AddStrings(FInitOrders);
end;

procedure TCnUsesInitTreeForm.UpdateTreeView;
var
  Node: TTreeNode;
  I: Integer;
  Leaf: TCnUsesLeaf;
begin
  tvTree.Items.Clear;
  Node := tvTree.Items.AddObject(nil,
    _CnExtractFileName(_CnChangeFileExt(FTree.Root.Text, '')), FTree.Root);

  FTree.SaveToTreeView(tvTree, Node);

  if chkSystemPath.Checked and chkProjectPath.Checked then
  begin
    if tvTree.Items.Count > 0 then
      tvTree.Items[0].Expand(True);

    statUses.SimpleText := Format(SCnBookmarkFileCount, [tvTree.Items.Count]);
    Exit;
  end;

  for I := tvTree.Items.Count - 1 downto 0 do
  begin
    Node := tvTree.Items[I];
    Leaf := TCnUsesLeaf(Node.Data);

    if not chkSystemPath.Checked and (Leaf.SearchType = mstSystemSearch) then
      tvTree.Items.Delete(Node)
    else if not chkProjectPath.Checked and (Leaf.SearchType = mstProjectSearch) then
      tvTree.Items.Delete(Node);
  end;

  if tvTree.Items.Count > 0 then
    tvTree.Items[0].Expand(True);

  statUses.SimpleText := Format(SCnBookmarkFileCount, [tvTree.Items.Count]);
end;

procedure TCnUsesInitTreeForm.chkSystemPathClick(Sender: TObject);
begin
  GenerateInitOrder;
  UpdateTreeView;
end;

procedure TCnUsesInitTreeForm.TreeSaveANode(ALeaf: TCnLeaf;
  ATreeNode: TTreeNode; var Valid: Boolean);
begin
  ATreeNode.Text := ALeaf.Text;
  ATreeNode.Data := ALeaf;
end;

procedure TCnUsesInitTreeForm.tvTreeChange(Sender: TObject;
  Node: TTreeNode);
var
  Leaf: TCnUsesLeaf;
begin
  if Node <> nil then
  begin
    Leaf := TCnUsesLeaf(Node.Data);
    if Leaf <> nil then
      UpdateInfo(Leaf);
  end;
end;

procedure TCnUsesInitTreeForm.UpdateInfo(Leaf: TCnLeaf);
var
  ALeaf: TCnUsesLeaf;
begin
  ALeaf := TCnUsesLeaf(Leaf);

  mmoSourceFileText.Lines.Text := ALeaf.SourceName;
  mmoDcuFileText.Lines.Text := ALeaf.DcuName;
  if ALeaf.SearchType <> mstInvalid then
    lblSearchTypeText.Caption := csSearchTypeStrings[ALeaf.SearchType]^
  else
    lblSearchTypeText.Caption := SCnUnknownNameResult;

  if ALeaf.IsImpl then
    lblUsesTypeText.Caption := 'implementation'
  else if not IsDpr(ALeaf.SourceName) and not IsDpk(ALeaf.SourceName) then
    lblUsesTypeText.Caption := 'interface'
  else
    lblUsesTypeText.Caption := '';
end;

procedure TCnUsesInitTreeForm.actExitExecute(Sender: TObject);
begin
  Close;
end;

procedure TCnUsesInitTreeForm.actHelpExecute(Sender: TObject);
begin
  ShowFormHelp;
end;

procedure TCnUsesInitTreeForm.actOpenExecute(Sender: TObject);
var
  Leaf: TCnUsesLeaf;
begin
  if tvTree.Selected <> nil then
  begin
    Leaf := TCnUsesLeaf(tvTree.Selected.Data);
    if (Leaf <> nil) and (Leaf.SourceName <> '') then
      CnOtaOpenFile(Leaf.SourceName);
  end;
end;

procedure TCnUsesInitTreeForm.actlstUsesUpdate(Action: TBasicAction;
  var Handled: Boolean);
begin
  if (Action = actOpen) or (Action = actLocateSource) then
    TCustomAction(Action).Enabled := tvTree.Selected <> nil
  else if (Action = actExport) or (Action = actSearch) or (Action = actSearchNext) then
    TCustomAction(Action).Enabled := tvTree.Items.Count > 1
  else if Action = actGenerateUsesTree then
    TCustomAction(Action).Enabled := cbbProject.Items.Count > 0;
end;

procedure TCnUsesInitTreeForm.actExportExecute(Sender: TObject);
var
  I: Integer;
  L: TStringList;
begin
  if dlgSave.Execute then
  begin
    L := TStringList.Create;
    try
      for I := 0 to tvTree.Items.Count - 1 do
      begin
        L.Add(Format('%2.2d:%s%s',[I + 1, StringOfChar(' ', tvTree.Items[I].Level),
          tvTree.Items[I].Text]));
      end;
      L.SaveToFile(_CnChangeFileExt(dlgSave.FileName, '.txt'));
    finally
      L.Free;
    end;
  end;
end;

procedure TCnUsesInitTreeForm.actSearchExecute(Sender: TObject);
begin
  if tvTree.Items.Count <= 0 then
    Exit;

  dlgFind.FindText := FOldSearchStr;
  dlgFind.Execute;
end;

procedure TCnUsesInitTreeForm.dlgFindClose(Sender: TObject);
begin
  FOldSearchStr := dlgFind.FindText;
end;

procedure TCnUsesInitTreeForm.dlgFindFind(Sender: TObject);
begin
  // ���� dlgFind.FindText �Լ�����ѡ����µȣ����� TreeView �ڵ� Node �� Text ����
  if not SearchText(dlgFind.FindText, frDown in dlgFind.Options,
    not (frMatchCase in dlgFind.Options), frWholeWord in dlgFind.Options) then
    ErrorDlg(SCnUsesInitTreeNotFound);
end;

function TCnUsesInitTreeForm.SearchText(const Text: string; ToDown,
  IgnoreCase, WholeWord: Boolean): Boolean;
var
  StartNode: TTreeNode;
  I, Idx, FindIdx: Integer;
  Found: Boolean;

  function MatchNode(ANode: TTreeNode): Boolean;
  var
    S1, S2: string;
  begin
    Result := False;
    if IgnoreCase then // ���Դ�Сд������Сд�Ƚ�
    begin
      S1 := LowerCase(Text);
      S2 := LowerCase(ANode.Text);
    end
    else // ƥ���Сд
    begin
      S1 := Text;
      S2 := ANode.Text;
    end;

    if WholeWord and (S1 = S2) then
      Result := True
    else if not WholeWord and (Pos(S1, S2) >= 1) then
      Result := True;
  end;

begin
  Result := False;
  StartNode := tvTree.Selected;
  if StartNode = nil then
  begin
    if ToDown then
      StartNode := tvTree.Items[0]
    else
      StartNode := tvTree.Items[tvTree.Items.Count - 1];
  end;

  if StartNode = nil then
    Exit;

  Idx := StartNode.AbsoluteIndex;
  Found := False;
  FindIdx := -1;

  if ToDown then
  begin
    for I := Idx + 1 to tvTree.Items.Count - 1 do
    begin
      if MatchNode(tvTree.Items[I]) then
      begin
        Found := True;
        FindIdx := I;
        Break;
      end;
    end;

    if not Found then
    begin
      for I := 0 to Idx do
      begin
        if MatchNode(tvTree.Items[I]) then
        begin
          Found := True;
          FindIdx := I;
          Break;
        end;
      end;
    end;
  end
  else
  begin
    for I := Idx - 1 downto 0 do
    begin
      if MatchNode(tvTree.Items[I]) then
      begin
        Found := True;
        FindIdx := I;
        Break;
      end;
    end;

    if not Found then
    begin
      for I := tvTree.Items.Count - 1 downto Idx do
      begin
        if MatchNode(tvTree.Items[I]) then
        begin
          Found := True;
          FindIdx := I;
          Break;
        end;
      end;
    end;
  end;

  if Found then
  begin
    tvTree.Selected := tvTree.Items[FindIdx];
    tvTree.Selected.MakeVisible;
    Result := True;
  end
  else
    Result := False;
end;

procedure TCnUsesInitTreeForm.actSearchNextExecute(Sender: TObject);
begin
  if FOldSearchStr = '' then
    dlgFind.Execute
  else if not SearchText(dlgFind.FindText, frDown in dlgFind.Options,
    not (frMatchCase in dlgFind.Options), frWholeWord in dlgFind.Options) then
    ErrorDlg(SCnUsesInitTreeNotFound);
end;

procedure TCnUsesInitTreeForm.actLocateSourceExecute(Sender: TObject);
var
  strExecute: string;
  Leaf: TCnUsesLeaf;
begin
  if tvTree.Selected = nil then
    Exit;

  Leaf := TCnUsesLeaf(tvTree.Selected.Data);
  if Leaf = nil then
    Exit;

  if FileExists(Leaf.SourceName) then
  begin
    strExecute := Format(csExploreCmdLine, [Leaf.SourceName]);
{$IFDEF UNICODE}
    WinExecute(strExecute, SW_SHOWNORMAL);
{$ELSE}
    WinExec(PAnsiChar(strExecute), SW_SHOWNORMAL);
{$ENDIF}
  end;
end;

function TCnUsesInitTreeForm.GetHelpTopic: string;
begin
  Result := 'CnUsesUnitsTools';
end;

{$ENDIF CNWIZARDS_CNUSESTOOLS}
end.

