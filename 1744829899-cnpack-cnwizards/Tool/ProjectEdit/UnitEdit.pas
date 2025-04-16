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

unit UnitEdit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls, FileCtrl, CnCommon, ComCtrls;

type
  TFormProjectEdit = class(TForm)
    PageControl1: TPageControl;
    tsCWProject: TTabSheet;
    lblCWRoot: TLabel;
    bvl1: TBevel;
    lblCWDpr: TLabel;
    lblCWDprAdd: TLabel;
    bvl2: TBevel;
    lblCWDproj: TLabel;
    lblCWDprojAdd: TLabel;
    bvl3: TBevel;
    lblCWBpf: TLabel;
    lblCWBpfAdd: TLabel;
    bvl4: TBevel;
    lblCWBpr: TLabel;
    lblCWBprAdd: TLabel;
    edtCWRootDir: TEdit;
    btnCWBrowse: TButton;
    edtCWDprBefore: TEdit;
    edtCWDprAdd: TEdit;
    btnCWDprAdd: TButton;
    btnCWDprojAdd: TButton;
    mmoCWDprojAdd: TMemo;
    mmoCWDprojBefore: TMemo;
    btnCWDprTemplate: TButton;
    edtCWBpfBefore: TEdit;
    edtCWBpfAdd: TEdit;
    btnCWBpfAdd: TButton;
    edtCWBprBefore: TEdit;
    edtCWBprAdd: TEdit;
    btnCWBprAdd: TButton;
    tsCVProject: TTabSheet;
    lblCVRoot: TLabel;
    edtCVRootDir: TEdit;
    btnCVBrowse: TButton;
    Bevel1: TBevel;
    edtCVDprBefore: TEdit;
    lblCVDpr: TLabel;
    edtCVDprAdd: TEdit;
    btnCVDprAdd: TButton;
    lblCVDprAdd: TLabel;
    tsCVSort: TTabSheet;
    lblCVSortRoot: TLabel;
    edtCVSortRootDir: TEdit;
    btnCVSortBrowse: TButton;
    btnCVSortDprAll: TButton;
    btnCVSortDprOne: TButton;
    dlgOpen1: TOpenDialog;
    btnCVSortDprAll1: TButton;
    bvl21: TBevel;
    lblCVDproj: TLabel;
    mmoCVDprojBefore: TMemo;
    mmoCVDprojAdd: TMemo;
    btnCVDprojAdd: TButton;
    lblCVDprojAdd: TLabel;
    bvl5: TBevel;
    btnCVSortDprojAll: TButton;
    btnCVSortDprojAll1: TButton;
    btnCVSortDprojOne: TButton;
    btnCVSortBpkOne: TButton;
    lbl1: TLabel;
    bvl211: TBevel;
    lblCVBpk: TLabel;
    edtCVBpkAdd: TEdit;
    edtCVBpkBefore: TEdit;
    btnCVBpkAdd: TButton;
    lblCVBpkAdd: TLabel;
    bvl6: TBevel;
    lblCVBpk1: TLabel;
    edtCVBpkBefore1: TEdit;
    edtCVBpkAdd1: TEdit;
    btnCVBpkAdd1: TButton;
    lblCVBpkAdd1: TLabel;
    lbl2: TLabel;
    procedure FormCreate(Sender: TObject);
    procedure btnCWBrowseClick(Sender: TObject);
    procedure btnCWDprAddClick(Sender: TObject);
    procedure btnCWDprojAddClick(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure btnCWBpfAddClick(Sender: TObject);
    procedure btnCWBprAddClick(Sender: TObject);
    procedure btnCVBrowseClick(Sender: TObject);
    procedure btnCVSortBrowseClick(Sender: TObject);
    procedure btnCVSortDprOneClick(Sender: TObject);
    procedure btnCVSortDprAllClick(Sender: TObject);
    procedure btnCVSortDprAll1Click(Sender: TObject);
    procedure btnCVDprAddClick(Sender: TObject);
    procedure btnCVSortDprojOneClick(Sender: TObject);
    procedure btnCVSortDprojAllClick(Sender: TObject);
    procedure btnCVSortDprojAll1Click(Sender: TObject);
    procedure btnCVDprojAddClick(Sender: TObject);
    procedure btnCVSortBpkOneClick(Sender: TObject);
    procedure btnCVBpkAddClick(Sender: TObject);
    procedure btnCVBpkAdd1Click(Sender: TObject);
  private
    FCount: Integer;
    FSingleBefore, FSingleAdd: string;
    FBefores, FAdds: TStrings;
    procedure SingleLineFound(const FileName: string; const Info: TSearchRec;
      var Abort: Boolean);
    procedure MultiLineFound(const FileName: string; const Info: TSearchRec;
      var Abort: Boolean);
  public
    procedure SortDprFileFound(const FileName: string; const Info: TSearchRec;
      var Abort: Boolean);
    procedure SortDprojFileFound(const FileName: string; const Info: TSearchRec;
      var Abort: Boolean);
    function SortOneDpr(const Dpr: string): Boolean;
    function SortOneDproj(const Proj: string): Boolean; // ����������ֻ�Ų���
    function SortOneBpk(const Bpk: string): Boolean; // �� obj �� <FILENAME ��
  end;

var
  FormProjectEdit: TFormProjectEdit;

implementation

{$R *.DFM}

{
  �����޸��������ݣ�
  CB5 BPF  - ��ʵ�� USEUNIT/USEFORMNS
  CB6 BPF  - ��ʵ�� USEFORMNS
  CB6 BPR  - ��ʵ�� <FILE FILENAME=
  DPR                 - ��ʵ��
  BDSPROJ/DPROJ       - ��ʵ��

  �� CB5/6 BPF �� obj �� dfm Ҫ�ֹ���
}

const
  FILE_COUNT = '�����ļ�����';
  FILE_OK = '�ļ��Ѵ���';

procedure TFormProjectEdit.FormCreate(Sender: TObject);
var
  S: string;
begin
  S := ExtractFileDir(Application.ExeName);
  S := ExtractFileDir(S);
  edtCWRootDir.Text := S + '\Source\';

  S := ExtractFileDir(S);
  edtCVRootDir.Text := S + '\cnvcl\Package\';
  edtCVSortRootDir.Text := edtCVRootDir.Text;

  FBefores := TStringList.Create;
  FAdds := TStringList.Create;
end;

procedure TFormProjectEdit.btnCWBrowseClick(Sender: TObject);
var
  S: string;
begin
  if SelectDirectory('Select CnPack IDE Wizards Project Files Directory', '', S) then
    edtCWRootDir.Text := S;
end;

procedure TFormProjectEdit.btnCWDprAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCWRootDir.Text) then
    Exit;

  if (Trim(edtCWDprBefore.Text) = '') or (Trim(edtCWDprAdd.Text) = '') then
    Exit;

  FCount := 0;
  FSingleBefore := Trim(edtCWDprBefore.Text);
  FSingleAdd := Trim(edtCWDprAdd.Text);
  FindFile(edtCWRootDir.Text, '*.dpr', SingleLineFound, nil, False, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.SingleLineFound(const FileName: string;
  const Info: TSearchRec; var Abort: Boolean);
var
  L: TStrings;
  I: Integer;
  S: string;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(FileName);
    for I := 0 to L.Count - 1 do
    begin
      if StrEndWith(L[I], FSingleBefore) then
      begin
        S := L[I];
        Delete(S, Pos(FSingleBefore, S), MaxInt);
        if Trim(S) = '' then
        begin
          // S ��Ŀ���е�ǰ���ո���
          L.Insert(I + 1, S + FSingleAdd);
          L.SaveToFile(FileName);
          Inc(FCount);
          Exit;
        end;
      end;
    end;
  finally
    L.Free;
  end;
end;

procedure TFormProjectEdit.btnCWDprojAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCWRootDir.Text) then
    Exit;

  if (Trim(mmoCWDprojBefore.Lines.Text) = '') or (Trim(mmoCWDprojAdd.Lines.Text) = '') then
    Exit;

  FCount := 0;
  FBefores.Assign(mmoCWDprojBefore.Lines);
  FAdds.Assign(mmoCWDprojAdd.Lines);

  if Trim(FBefores[FBefores.Count - 1]) = '' then
    FBefores.Delete(FBefores.Count - 1);
  if Trim(FAdds[FAdds.Count - 1]) = '' then
    FAdds.Delete(FAdds.Count - 1);

  FindFile(edtCWRootDir.Text, '*.*proj', MultiLineFound, nil, False, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.MultiLineFound(const FileName: string;
  const Info: TSearchRec; var Abort: Boolean);
var
  L: TStringList;
  I, K: Integer;
  S: string;
  IsTab: Boolean;

  function StringsMatch(SourceStartIndex: Integer; Source, Patts: TStrings): Boolean;
  var
    J: Integer;
  begin
    // �ж� Source �ĵ� SourceStartIndex �����Ƿ�ƥ�� Patts ��������
    Result := False;
    if SourceStartIndex > Source.Count - Patts.Count then
      Exit; // �����ȵ�

    for J := 0 to Patts.Count - 1 do
    begin
      if Trim(Source[SourceStartIndex + J]) <> Trim(Patts[J]) then
        Exit;
    end;
    Result := True;
  end;

  procedure PutToList(List: TStringList; FoundPos: Integer; const Str: string);
  begin
    if FoundPos >= List.Count then
      List.Add(Str)
    else
    begin
      if FoundPos < 0 then
        FoundPos := 0;
      List.Insert(FoundPos + 1, Str);
    end;
  end;

begin
  L := TStringList.Create;
  try
    L.LoadFromFile(FileName);

    for I := 0 to L.Count - 1 do
    begin
      if StringsMatch(I, L, FBefores) then
      begin
        S := L[I + FBefores.Count - 1];
        Delete(S, Pos(FBefores[FBefores.Count - 1], S), MaxInt);
        // S ��Ŀ���е�һ�е�ǰ���ո����� Tab ��

        IsTab := (Length(S) > 0) and (S[1] = #9);

        for K := FAdds.Count - 1 downto 0 do // �ݲ�֧�����
        begin // ��������
          if IsTab and (Length(FAdds[K]) > 0) and (FAdds[K][1] = ' ') then
          begin
            PutToList(L, I + FBefores.Count - 1, S + #9 + Trim(FAdds[K]));
          end
          else if not IsTab and (Length(FAdds[K]) > 0) and (FAdds[K][1] = ' ') then
          begin
            PutToList(L, I + FBefores.Count - 1, S + '    ' + Trim(FAdds[K])); // û���жϼ����ո�ֻ�������ĸ�����
          end
          else
            PutToList(L, I + FBefores.Count - 1, S + FAdds[K]);
        end;

        L.SaveToFile(FileName);
        Inc(FCount);
        Exit;
      end;
    end;
  finally
    L.Free;
  end;
end;

procedure TFormProjectEdit.FormDestroy(Sender: TObject);
begin
  FBefores.Free;
  FAdds.Free;
end;

procedure TFormProjectEdit.btnCWBpfAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCWRootDir.Text) then
    Exit;

  if (Trim(edtCWBpfBefore.Text) = '') or (Trim(edtCWBpfAdd.Text) = '') then
    Exit;

  FCount := 0;
  FSingleBefore := Trim(edtCWBpfBefore.Text);
  FSingleAdd := Trim(edtCWBpfAdd.Text);
  FindFile(edtCWRootDir.Text, '*.bpf', SingleLineFound, nil, False, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCWBprAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCWRootDir.Text) then
    Exit;

  if (Trim(edtCWBprBefore.Text) = '') or (Trim(edtCWBprAdd.Text) = '') then
    Exit;

  FCount := 0;
  FSingleBefore := Trim(edtCWBprBefore.Text);
  FSingleAdd := Trim(edtCWBprAdd.Text);
  FindFile(edtCWRootDir.Text, '*.bpr', SingleLineFound, nil, False, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVBrowseClick(Sender: TObject);
var
  S: string;
begin
  if SelectDirectory('Select CnPack Component Project Files Directory', '', S) then
    edtCVRootDir.Text := S;
end;

procedure TFormProjectEdit.btnCVSortBrowseClick(Sender: TObject);
var
  S: string;
begin
  if SelectDirectory('Select CnPack Component Project Files Directory', '', S) then
    edtCVRootDir.Text := S;
end;

procedure TFormProjectEdit.btnCVSortDprOneClick(Sender: TObject);
begin
  if dlgOpen1.Execute then
  begin
    SortOneDpr(dlgOpen1.FileName);
    ShowMessage(FILE_OK + dlgOpen1.FileName);
  end;
end;

function DprCompare(List: TStringList; Index1, Index2: Integer): Integer;
const
  IN_KEYWORD = ' in ';
var
  P: Integer;
  S1, S2: string;
  F1, D1: string;
  F2, D2: string;
begin
  // �������ļ��������һ����Ŀ¼���������ź�����ǰ
  S1 := Trim(List[Index1]);
  S2 := Trim(List[Index2]);

  // �õ��ļ���
  P := Pos(IN_KEYWORD, S1);
  if P > 1 then
  begin
    F1 := Copy(S1, 1, P - 1);
    Delete(S1, 1, P + Length(IN_KEYWORD));
  end;

  P := Pos(IN_KEYWORD, S2);
  if P > 1 then
  begin
    F2 := Copy(S2, 1, P - 1);
    Delete(S2, 1, P + Length(IN_KEYWORD));
  end;

  // ȥ��·�����ĵ�һ��������
  if S1[1] = '''' then
    Delete(S1, 1, 1);
  if S2[1] = '''' then
    Delete(S2, 1, 1);

  // ȥ�����һ�� ', �� '; �Լ�����Ĳ���
  if StrEndWith(S1, ''',') or StrEndWith(S1, ''';') then
    Delete(S1, Length(S1) - 1, MaxInt);
  if StrEndWith(S2, ''',') or StrEndWith(S2, ''';') then
    Delete(S2, Length(S2) - 1, MaxInt);

  // ��ȥ�����һ���������Լ�����Ĳ��֣������д���ע�ʹ���
  P := LastCharPos(S1, '''');
  if P > 0 then
    Delete(S1, P, MaxInt);
  P := LastCharPos(S2, '''');
  if P > 0 then
    Delete(S2, P, MaxInt);

  D1 := ExtractFilePath(S1);
  if StrEndWith(D1, '\') then
    Delete(D1, Length(D1), 1);
  D1 := ExtractFileName(D1);

  D2 := ExtractFilePath(S2);
  if StrEndWith(D2, '\') then
    Delete(D2, Length(D2), 1);
  D2 := ExtractFileName(D2);

  Result := CompareStr(UpperCase(D1), UpperCase(D2));
  if Result = 0 then
    Result := CompareStr(UpperCase(F1), UpperCase(F2));
end;

function TFormProjectEdit.SortOneDpr(const Dpr: string): Boolean;
var
  I, F1, F2: Integer;
  L1, L2: TStringList;
begin
  Result := False;
  L1 := nil;
  L2 := nil;

  try
    L1 := TStringList.Create;
    L1.LoadFromFile(Dpr);

    F1 := -1;
    F2 := -1;
    for I := 0 to L1.Count - 1 do
    begin
      if Trim(L1[I]) = 'contains' then
        F1 := I + 1;
      if Trim(L1[I]) = 'end.'then
        F2 := I - 2;
    end;

    if (F1 > 1) and (F2 > F1) then
    begin
      L2 := TStringList.Create;
      for I := F1 to F2 do
        L2.Add(L1[I]);

      L2.CustomSort(DprCompare);
      for I := F1 to F2 do
        L1[I] := L2[I - F1];

      L1.SaveToFile(Dpr);
      Result := True;
    end;
  finally
    L2.Free;
    L1.Free;
  end;
end;

procedure TFormProjectEdit.btnCVSortDprAllClick(Sender: TObject);
begin
  if not DirectoryExists(edtCVSortRootDir.Text) then
    Exit;

  FCount := 0;
  FindFile(edtCVSortRootDir.Text, 'CnPack*.dpk', SortDprFileFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.SortDprFileFound(const FileName: string;
  const Info: TSearchRec; var Abort: Boolean);
begin
  if SortOneDpr(FileName) then
    Inc(FCount);
end;

procedure TFormProjectEdit.btnCVSortDprAll1Click(Sender: TObject);
begin
  if not DirectoryExists(edtCVSortRootDir.Text) then
    Exit;

  FCount := 0;
  FindFile(edtCVSortRootDir.Text, 'dclCnPack*.dpk', SortDprFileFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVDprAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCVRootDir.Text) then
    Exit;

  if (Trim(edtCVDprBefore.Text) = '') or (Trim(edtCVDprAdd.Text) = '') then
    Exit;

  FCount := 0;
  FSingleBefore := Trim(edtCVDprBefore.Text);
  FSingleAdd := Trim(edtCVDprAdd.Text);
  FindFile(edtCVRootDir.Text, '*.dpk', SingleLineFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVSortDprojOneClick(Sender: TObject);
begin
  if dlgOpen1.Execute then
  begin
    SortOneDproj(dlgOpen1.FileName);
    ShowMessage(FILE_OK + dlgOpen1.FileName);
  end;
end;

function SimpleCompare(List: TStringList; Index1, Index2: Integer): Integer;
begin
  Result := CompareStr(UpperCase(Trim(List[Index1])), UpperCase(Trim(List[Index2])));
end;

function TFormProjectEdit.SortOneDproj(const Proj: string): Boolean;
var
  I, F1, F2: Integer;
  L1, L2: TStringList;
begin
  Result := False;
  L1 := nil;
  L2 := nil;

  try
    L1 := TStringList.Create;
    L1.LoadFromFile(Proj);

    F1 := -1;
    F2 := -1;
    for I := 0 to L1.Count - 1 do
    begin
      if (F1 < 0) and (Pos('<DCCReference Include="..\..\', Trim(L1[I])) = 1) then
        F1 := I;
      if (F1 > 0) and (F2 < 0) and StrEndWith(Trim(L1[I]), '">') then // ������ Form��ֻ������ǰ��
        F2 := I - 1;
    end;

    if (F1 > 1) and (F2 > F1) then
    begin
      L2 := TStringList.Create;
      for I := F1 to F2 do
        L2.Add(L1[I]);

      L2.CustomSort(SimpleCompare);
      for I := F1 to F2 do
        L1[I] := L2[I - F1];

      L1.SaveToFile(Proj);
      Result := True;
    end;
  finally
    L2.Free;
    L1.Free;
  end;
end;

procedure TFormProjectEdit.SortDprojFileFound(const FileName: string;
  const Info: TSearchRec; var Abort: Boolean);
begin
  if SortOneDproj(FileName) then
    Inc(FCount);
end;

procedure TFormProjectEdit.btnCVSortDprojAllClick(Sender: TObject);
begin
  if not DirectoryExists(edtCVSortRootDir.Text) then
    Exit;

  FCount := 0;
  FindFile(edtCVSortRootDir.Text, 'CnPack*.*proj', SortDprojFileFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVSortDprojAll1Click(Sender: TObject);
begin
  if not DirectoryExists(edtCVSortRootDir.Text) then
    Exit;

  FCount := 0;
  FindFile(edtCVSortRootDir.Text, 'dclCnPack*.*proj', SortDprojFileFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVDprojAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCVRootDir.Text) then
    Exit;

  if (Trim(mmoCVDprojBefore.Lines.Text) = '') or (Trim(mmoCVDprojAdd.Lines.Text) = '') then
    Exit;

  FCount := 0;
  FBefores.Assign(mmoCVDprojBefore.Lines);
  FAdds.Assign(mmoCVDprojAdd.Lines);

  if Trim(FBefores[FBefores.Count - 1]) = '' then
    FBefores.Delete(FBefores.Count - 1);
  if Trim(FAdds[FAdds.Count - 1]) = '' then
    FAdds.Delete(FAdds.Count - 1);

  FindFile(edtCVRootDir.Text, '*.*proj', MultiLineFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVSortBpkOneClick(Sender: TObject);
begin
  if dlgOpen1.Execute then
  begin
    SortOneBpk(dlgOpen1.FileName);
    ShowMessage(FILE_OK + dlgOpen1.FileName);
  end;
end;

function TFormProjectEdit.SortOneBpk(const Bpk: string): Boolean;
var
  I, F1, F2: Integer;
  L1, L2: TStringList;
  Found: Boolean;
begin
  Result := False;
  L1 := nil;
  L2 := nil;
  Found := False;

  try
    L1 := TStringList.Create;
    L1.LoadFromFile(Bpk);

    // �� bpk �е� obj ����
    F1 := -1;
    F2 := -1;
    for I := 0 to L1.Count - 1 do
    begin
      if (F1 < 0) and (Pos('..\..\Source\', Trim(L1[I])) = 1) and StrEndWith(Trim(L1[I]), '.obj') then
        F1 := I;
      if (F1 > 0) and (F2 < 0) and StrEndWith(Trim(L1[I]), '"/>') then
        F2 := I - 1;
    end;

    if (F1 > 1) and (F2 > F1) then
    begin
      L2 := TStringList.Create;
      for I := F1 to F2 do
        L2.Add(L1[I]);

      L2.CustomSort(SimpleCompare);
      for I := F1 to F2 do
        L1[I] := L2[I - F1];

      Found := True;
    end;

    // �� <FILENAME ����
    F1 := -1;
    F2 := -1;
    for I := 0 to L1.Count - 1 do
    begin
      if (F1 < 0) and (Pos('<FILE FILENAME="..\..\', Trim(L1[I])) = 1) then
        F1 := I;
      if (F1 > 0) and (F2 < 0) and (Pos('<FILE FILENAME="', Trim(L1[I])) = 1)
        and (Pos('<FILE FILENAME="..\..\', Trim(L1[I])) <> 1) then
        F2 := I - 1;
    end;

    if (F1 > 1) and (F2 > F1) then
    begin
      L2 := TStringList.Create;
      for I := F1 to F2 do
        L2.Add(L1[I]);

      L2.CustomSort(SimpleCompare);
      for I := F1 to F2 do
        L1[I] := L2[I - F1];

      Found := True;
    end;

    if Found then
    begin
      L1.SaveToFile(Bpk);
      Result := True;
    end;
  finally
    L2.Free;
    L1.Free;
  end;
end;

procedure TFormProjectEdit.btnCVBpkAddClick(Sender: TObject);
begin
  if not DirectoryExists(edtCVRootDir.Text) then
    Exit;

  if (Trim(edtCVBpkBefore.Text) = '') or (Trim(edtCVBpkAdd.Text) = '') then
    Exit;

  FCount := 0;
  FSingleBefore := Trim(edtCVBpkBefore.Text) + ' ';
  FSingleAdd := Trim(edtCVBpkAdd.Text) + ' '; // obj �ļ�����һ���ո�
  FindFile(edtCVRootDir.Text, '*.bpk', SingleLineFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

procedure TFormProjectEdit.btnCVBpkAdd1Click(Sender: TObject);
begin
  if not DirectoryExists(edtCVRootDir.Text) then
    Exit;

  if (Trim(edtCVBpkBefore1.Text) = '') or (Trim(edtCVBpkAdd1.Text) = '') then
    Exit;

  FCount := 0;
  FSingleBefore := Trim(edtCVBpkBefore1.Text);
  FSingleAdd := Trim(edtCVBpkAdd1.Text); // obj �ļ�����һ���ո�
  FindFile(edtCVRootDir.Text, '*.bpk', SingleLineFound, nil, True, False);

  if FCount > 0 then
    InfoDlg(FILE_COUNT + IntToStr(FCount));
end;

end.
