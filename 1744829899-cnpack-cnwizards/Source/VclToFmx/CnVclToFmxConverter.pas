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

unit CnVclToFmxConverter;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�CnWizards VCL/FMX ����ת������Ԫ
* ��Ԫ���ߣ�CnPack ������ (master@cnpack.org)
* ��    ע���õ�Ԫ�� Delphi 10.3.1 �� VCL �� FMX Ϊ����ȷ����һЩӳ���ϵ
* ����ƽ̨��PWin7 + Delphi 10.3.1
* ���ݲ��ԣ�XE2 �����ϣ���֧�ָ��Ͱ汾
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2019.04.10 V1.0
*               ������Ԫ��ʵ�ֻ�������
================================================================================
|</PRE>}

interface

{$I CnPack.inc}

uses
  System.SysUtils, System.Classes, System.Generics.Collections, Winapi.Windows,
  FMX.Graphics, Vcl.ComCtrls, Vcl.Graphics, Vcl.Imaging.jpeg, Vcl.Imaging.pngimage,
  Vcl.Imaging.GIFImg,  Vcl.Controls, System.TypInfo,
  CnFmxUtils, CnVclToFmxMap, CnWizDfmParser, CnStrings, CnCommon;

type
  // === ����ת���� ===

  TCnPositionConverter = class(TCnPropertyConverter)
  {* �� Left/Top ת���� Position ���Ե�ת����}
  public
    class procedure GetProperties(OutProperties: TStrings); override;
    class procedure ProcessProperties(const PropertyName, TheClassName,
      PropertyValue: string; InProperties, OutProperties: TStrings;
      Tab: Integer = 0); override;
  end;

  TCnSizeConverter = class(TCnPropertyConverter)
  {* �� Width/Height ת���� Size ���Ե�ת����}
  public
    class procedure GetProperties(OutProperties: TStrings); override;
    class procedure ProcessProperties(const PropertyName, TheClassName,
      PropertyValue: string; InProperties, OutProperties: TStrings;
      Tab: Integer = 0); override;
  end;

  TCnCaptionConverter = class(TCnPropertyConverter)
  {* �� Caption ת���� Text ���Ե�ת����}
  public
    class procedure GetProperties(OutProperties: TStrings); override;
    class procedure ProcessProperties(const PropertyName, TheClassName,
      PropertyValue: string; InProperties, OutProperties: TStrings;
      Tab: Integer = 0); override;
  end;

  TCnFontConverter = class(TCnPropertyConverter)
  {* �� Font ת���� TextSettings ���Ե�ת����}
  public
    class procedure GetProperties(OutProperties: TStrings); override;
    class procedure ProcessProperties(const PropertyName, TheClassName,
      PropertyValue: string; InProperties, OutProperties: TStrings;
      Tab: Integer = 0); override;
  end;

  TCnTouchConverter = class(TCnPropertyConverter)
  {* ת�� Touch ���Ե�ת����}
  public
    class procedure GetProperties(OutProperties: TStrings); override;
    class procedure ProcessProperties(const PropertyName, TheClassName,
      PropertyValue: string; InProperties, OutProperties: TStrings;
      Tab: Integer = 0); override;
  end;

  TCnGeneralConverter = class(TCnPropertyConverter)
  {* ת��һЩ��ͨ���Ե�ת����}
  public
    class procedure GetProperties(OutProperties: TStrings); override;
    class procedure ProcessProperties(const PropertyName, TheClassName,
      PropertyValue: string; InProperties, OutProperties: TStrings;
      Tab: Integer = 0); override;
  end;

  // === ���ת���� ===

  TCnTreeViewConverter = class(TCnComponentConverter)
  {* �����ת�� TreeView �����ת����}
  private
    class procedure LoadTreeLeafFromStream(Root, Leaf: TCnDfmLeaf; Stream: TStream);
  public
    class procedure GetComponents(OutVclComponents: TStrings); override;
    class procedure ProcessComponents(SourceLeaf, DestLeaf: TCnDfmLeaf; Tab: Integer = 0); override;
  end;

  TCnImageConverter = class(TCnComponentConverter)
  {* �����ת�� Image �����ת��������Ҫ������ Picture.Data ����}
  public
    class procedure GetComponents(OutVclComponents: TStrings); override;
    class procedure ProcessComponents(SourceLeaf, DestLeaf: TCnDfmLeaf; Tab: Integer = 0); override;
  end;

  TCnGridConverter = class(TCnComponentConverter)
  {* �����ת�� Grid �����ת��������Ҫ������ Options �����Լ���}
  public
    class procedure GetComponents(OutVclComponents: TStrings); override;
    class procedure ProcessComponents(SourceLeaf, DestLeaf: TCnDfmLeaf; Tab: Integer = 0); override;
  end;

  TCnImageListConverter = class(TCnComponentConverter)
  {* �����ת�� ImageList �����ת��������Ҫ������ Bitmap ����}
  public
    class procedure GetComponents(OutVclComponents: TStrings); override;
    class procedure ProcessComponents(SourceLeaf, DestLeaf: TCnDfmLeaf; Tab: Integer = 0); override;
  end;

  TCnToolBarConverter = class(TCnComponentConverter)
  {* �����ת�� ToolBar �����ת��������Ҫ������ Images ���Ը��ڲ��� Button}
  public
    class procedure GetComponents(OutVclComponents: TStrings); override;
    class procedure ProcessComponents(SourceLeaf, DestLeaf: TCnDfmLeaf; Tab: Integer = 0); override;
  end;

function CnVclToFmxConvert(const InDfmFile: string; var OutFmx, OutPas: string; const NewFileName: string = ''): Boolean;
{* ת������ڣ����� dfm����� fmx �ļ����ݣ�����ж�Ӧ�� Vcl �� Pas����Ҳת�������Ӧ FMX �� Pas �ļ�����
  ���� NewFileName �ɴ��գ��� OutPas �����е� UnitName δ��}

function CnVclToFmxReplaceUnitName(const PasName, InPas: string): string;
{* ���� CnConvertVclToFmx ת���õ��� OutPas����� NewFileName �ǿգ����ڲ� UnitName δ��
  ���ȷ�������ļ������滻 InPas �е���ʵ�ļ���}

function CnVclToFmxSaveContent(const FileName, InFile: string; SourceUtf8: Boolean = False): Boolean;
{* ���ļ����ݱ������ļ���fmx �ļ�Ĭ�� Ansi ���룬�ڲ��������� Unicode ת��
  Դ���ļ�Ĭ�� Utf8 ����}

implementation

uses
  mPasLex, CnPasWideLex, CnWidePasParser;

const
  UNIT_NAME_TAG = '<@#UnitName@>';

  UNIT_NAMES_PREFIX: array[0..5] of string = (
    'Graphics', 'Controls', 'Forms', 'Dialogs', 'StdCtrls', 'ExtCtrls'
  );

  UNIT_NAMES_DELETE: array[0..4] of string = (
    'ComCtrls', 'Buttons', 'CheckLst', 'FileCtrl', 'Vcl.Imaging.pngimage'
  );

  FMX_PURE_UNIT_PAIRS: array[0..3] of string = (
    'Clipbrd:FMX.Clipboard',
    'Sample.Spin:FMX.SpinBox',
    'Spin:FMX.SpinBox',
    'Grids:FMX.Grid'
    // 'Vcl.Clipbrd:FMX.Clipboard' // ���� Vcl ǰ׺�������滻����
  );

  CRLF = #13#10;

type
  TVclGraphicAccess = class(Vcl.Graphics.TGraphic);

  TVclImageListAccess = class(Vcl.Controls.TImageList);

// �� [a, b] ���ּ�����ʽ���ַ���ת��Ϊ����� a b ����Ԫ���ַ���
procedure ConvertSetStringToElements(const SetString: string; OutElements: TStringList);
var
  S: string;
begin
  S := SetString;
  if Length(S) <= 2 then
    Exit;

  if S[1] = '[' then
    Delete(S, 1, 1);
  if S[Length(S)] = ']' then
    Delete(S, Length(S), 1);

  S := StringReplace(S, ' ', '', [rfReplaceAll]);
  OutElements.CommaText := S;
end;

// �� a b ���ַ���ļ���Ԫ���ַ�����ϳ� [a, b] ���ּ�����ʽ���ַ���
function ConvertSetElementsToString(InElements: TStrings): string;
var
  I: Integer;
begin
  if (InElements = nil) or (InElements.Count = 0) then
  begin
    Result := '[]';
    Exit;
  end;

  Result := '[' + InElements.CommaText + ']';
  Result := StringReplace(Result, ',', ', ', [rfReplaceAll]);
end;

// �����Ϊͨ�õ� Vcl. ����ǰ׺��Ԫ�� FMX. ���õ�Ԫ��ת��
function ReplaceVclUsesToFmx(const OldUses: string; UsesList: TStrings): string;
var
  I, L: Integer;
  OS, NS: string;
begin
  Result := OldUses;

  // ɾ�������� FMX ����ͬ����Ԫ�ĳ��ϡ������Ӧ���µĲ�ͬ����Ԫ�����ɺ�������ӳ���������
  for I := Low(UNIT_NAMES_DELETE) to High(UNIT_NAMES_DELETE) do
  begin
    Result := CnStringReplace(Result, UNIT_NAMES_DELETE[I] + ',', '', [crfIgnoreCase, crfReplaceAll, crfWholeWord]);
    Result := CnStringReplace(Result, UNIT_NAMES_DELETE[I], '', [crfIgnoreCase, crfReplaceAll, crfWholeWord]);
  end;

  // �Ȱ��� Vcl ǰ׺��ͳͳ�滻�ɴ� FMX ǰ׺��
  Result := StringReplace(Result, ' Vcl.', ' FMX.', [rfIgnoreCase, rfReplaceAll]);
  Result := StringReplace(Result, ',Vcl.', ', FMX.', [rfIgnoreCase, rfReplaceAll]);

  // �ٰ�ָ���� Vcl ǰ׺���滻�ɴ� FMX ǰ׺��
  for I := Low(UNIT_NAMES_PREFIX) to High(UNIT_NAMES_PREFIX) do
  begin
    Result := StringReplace(Result, ' ' + UNIT_NAMES_PREFIX[I], ' FMX.' + UNIT_NAMES_PREFIX[I], [rfIgnoreCase, rfReplaceAll]);
    Result := StringReplace(Result, ',' + UNIT_NAMES_PREFIX[I], ', FMX.' + UNIT_NAMES_PREFIX[I], [rfIgnoreCase, rfReplaceAll]);
  end;

  // �����滻��������ĸ����˵ĵ�Ԫ������ 'Clipbrd' �滻Ϊ 'FMX.Clipboard'
  for I := Low(FMX_PURE_UNIT_PAIRS) to High(FMX_PURE_UNIT_PAIRS) do
  begin
    L := Pos(':', FMX_PURE_UNIT_PAIRS[I]);
    if L > 0 then
    begin
      OS := Copy(FMX_PURE_UNIT_PAIRS[I], 1, L - 1);
      NS := Copy(FMX_PURE_UNIT_PAIRS[I], L + 1, MaxInt);
      if (OS <> '') and (NS <> '') then
        Result := CnStringReplace(Result, OS, NS, [crfReplaceAll, crfIgnoreCase, crfWholeWord]);
    end;
  end;

  // ��� Result ĩβ�Ƕ��Ż��߶��żӿո�����Ҫȥ��
  if Length(Result) > 1 then
  begin
    if Result[Length(Result)] = ',' then
      Delete(Result, Length(Result), 1)
    else if (Result[Length(Result)] = ' ') and (Result[Length(Result) - 1] = ',') then
      Delete(Result, Length(Result) - 1, 2);
  end;

  // �ٰ������ĺϲ���ȥ
  for I := 0 to UsesList.Count - 1 do
  begin
    if Pos(UsesList[I], Result) <= 0 then
      Result := Result + ', ' + UsesList[I];
  end;
end;

// ���ת����� fmx �� Tree �����⴦����� TabControl �� Sizes ����ֵ
procedure HandleTreeSpecial(ATree: TCnDfmTree);
var
  I, J: Integer;
  W, H, S: string;
  Leaf: TCnDfmLeaf;

  function ConvertToSingle(const V: string): string;
  begin
    Result := V;
    if Pos('.', V) > 0 then
      Result := Copy(V, 1, Pos('.', V) - 1) + 's';
  end;

begin
  for I := 0 to ATree.Count - 1 do
  begin
    Leaf := ATree.Items[I];
    if (Leaf.ElementClass = 'TTabControl') and (Leaf.Count > 0) then
    begin
      // ������ Item �� TTabControl���� Width �� Height
      W := Leaf.PropertyValue['Size.Width'];
      H := Leaf.PropertyValue['Size.Height'];
      W := ConvertToSingle(W);
      H := ConvertToSingle(H);

      if (W <> '') and (H <> '') then
      begin
        S := 'Sizes = (';
        for J := 0 to Leaf.Count - 1 do
        begin
          S := S + CRLF + '  ' + W;
          if J = Leaf.Count - 1 then
            S := S + CRLF + '  ' + H + ')'
          else
            S := S + CRLF + '  ' + H;
        end;
        Leaf.Properties.Add(S);
      end;
    end;
  end;
end;

function SearchPropertyValueAndRemoveFromStrings(List: TStrings; const PropertyName: string): string;
var
  I, P: Integer;
  S: string;
begin
  Result := '';
  S := PropertyName + ' = ';
  for I := List.Count - 1 downto 0 do
  begin
    P := Pos(S, List[I]);
    if P = 1 then
    begin
      Result := Copy(List[I], Length(S) + 1, MaxInt);
      List.Delete(I);
      Exit;
    end;
  end;
end;

{ TCnPositionConverter }

class procedure TCnPositionConverter.GetProperties(OutProperties: TStrings);
begin
  if OutProperties <> nil then
  begin
    OutProperties.Add('Top');
    OutProperties.Add('Left');
  end;
end;

class procedure TCnPositionConverter.ProcessProperties(const PropertyName,
  TheClassName, PropertyValue: string; InProperties, OutProperties: TStrings;
  Tab: Integer);
var
  X, Y: Integer;
  V, OutClassName: string;
  Cls: TClass;
begin
  OutClassName := CnGetFmxClassFromVclClass(TheClassName, InProperties);
  if not CnIsSupportFMXControl(OutClassName) then
  begin
    // Ŀ���಻�� FMX.TControl �����ֱ࣬��ʹ��ԭʼ Left/Top������ Position.X/Y
    OutProperties.Add(Format('%s = %s', [PropertyName, PropertyValue]));
  end
  else
  begin
    if PropertyName = 'Top' then
    begin
      Y := StrToIntDef(PropertyValue, 0);
      V := SearchPropertyValueAndRemoveFromStrings(InProperties, 'Left');
      X := StrToIntDef(V, 0);
    end
    else if PropertyName = 'Left' then
    begin
      X := StrToIntDef(PropertyValue, 0);
      V := SearchPropertyValueAndRemoveFromStrings(InProperties, 'Top');
      Y := StrToIntDef(V, 0);
    end
    else
      Exit;

    OutProperties.Add('Position.X = ' + GetFloatStringFromInteger(X));
    OutProperties.Add('Position.Y = ' + GetFloatStringFromInteger(Y));
  end;
end;

{ TCnTextConverter }

class procedure TCnCaptionConverter.GetProperties(OutProperties: TStrings);
begin
  if OutProperties <> nil then
    OutProperties.Add('Caption');
end;

class procedure TCnCaptionConverter.ProcessProperties(const PropertyName,
  TheClassName, PropertyValue: string; InProperties, OutProperties: TStrings;
  Tab: Integer);
begin
  // FMX TPanel / TToolBar ��Ӧ�� TGridLayout û�� Text ����
  if (PropertyName = 'Caption') and ((TheClassName <> 'TPanel') and
    (TheClassName <> 'TToolBar')) then
    OutProperties.Add('Text = ' + PropertyValue);

  // ToolButton ��Ӧ�� SpeedButton ��ǰû�� Text ���Ժ���������
end;

{ TCnSizeConverter }

class procedure TCnSizeConverter.GetProperties(OutProperties: TStrings);
begin
  if OutProperties <> nil then
  begin
    OutProperties.Add('Width');
    OutProperties.Add('Height');
  end;
end;

class procedure TCnSizeConverter.ProcessProperties(const PropertyName,
  TheClassName, PropertyValue: string; InProperties, OutProperties: TStrings;
  Tab: Integer);
var
  W, H: Integer;
  V: string;
begin
  if PropertyName = 'Width' then
  begin
    W := StrToIntDef(PropertyValue, 0);
    V := SearchPropertyValueAndRemoveFromStrings(InProperties, 'Height');
    H := StrToIntDef(V, 0);
  end
  else if PropertyName = 'Height' then
  begin
    H := StrToIntDef(PropertyValue, 0);
    V := SearchPropertyValueAndRemoveFromStrings(InProperties, 'Width');
    W := StrToIntDef(V, 0);
  end
  else
    Exit;

  OutProperties.Add('Size.Width = ' + GetFloatStringFromInteger(W));
  OutProperties.Add('Size.Height = ' + GetFloatStringFromInteger(H));
end;

{ TCnFontConverter }

class procedure TCnFontConverter.GetProperties(OutProperties: TStrings);
begin
  if OutProperties <> nil then
  begin
    OutProperties.Add('Font.Charset'); // û��
    OutProperties.Add('Font.Color');   // TextSettings.FontColor
    OutProperties.Add('Font.Height');  // TextSettings.Font.Size���Ӹ��������㷨���о�
    OutProperties.Add('Font.Name');    // TextSettings.Font.Family
    OutProperties.Add('Font.Style');   // TextSettings.Font.StyleExt������������ת��������о�
    OutProperties.Add('WordWrap');     // TextSettings.WordWrap
  end;
end;

class procedure TCnFontConverter.ProcessProperties(const PropertyName,
  TheClassName, PropertyValue: string; InProperties, OutProperties: TStrings;
  Tab: Integer);
var
  V, ScreenLogPixels: Integer;
  DC: HDC;
  NewStr: string;
begin
  if PropertyName = 'Font.Charset' then
    // ɶ���������������Ҳ�����Ӧ��
  else if PropertyName = 'Font.Color' then
  begin
    NewStr := CnConvertEnumValue(PropertyValue);
    if Length(NewStr) > 0 then
    begin
      if NewStr[1] in ['A'..'Z'] then // TextSettings �� FontColor ֵ����ɫ����ǰ��Ҫ�� cla
        NewStr := 'cla' + NewStr;
      OutProperties.Add('TextSettings.FontColor = ' + NewStr);
    end;
  end
  else if PropertyName = 'Font.Name' then
    OutProperties.Add('TextSettings.Font.Family = ' + PropertyValue)
  else if PropertyName = 'Font.Height' then
  begin
    // ���� Height ���� Size ��ֵ
    V := StrToIntDef(PropertyValue, -11);
    DC := GetDC(0);
    ScreenLogPixels := GetDeviceCaps(DC, LOGPIXELSY);
    ReleaseDC(0, DC);
    V := -MulDiv(V, 72, ScreenLogPixels);
    OutProperties.Add('TextSettings.Font.Size = ' + GetFloatStringFromInteger(V));
  end
  else if PropertyName = 'Font.Style' then
  begin
    // TODO: ���� StyleExt �Ķ�����ֵ
    OutProperties.Add('TextSettings.Font.StyleExt = ');
  end
  else if PropertyName = 'WordWrap' then
    OutProperties.Add('TextSettings.WordWrap = ' + PropertyValue);
end;

{ TCnTouchConverter }

class procedure TCnTouchConverter.GetProperties(OutProperties: TStrings);
begin
  if OutProperties <> nil then
    OutProperties.Add('Touch.');
end;

class procedure TCnTouchConverter.ProcessProperties(const PropertyName,
  TheClassName, PropertyValue: string; InProperties, OutProperties: TStrings;
  Tab: Integer);
begin
  if Pos('Touch.', PropertyName) = 1 then
    OutProperties.Add(Format('%s = %s', [PropertyName, PropertyValue]));
end;

{ TCnGeneralConverter }

class procedure TCnGeneralConverter.GetProperties(OutProperties: TStrings);
begin
  if OutProperties <> nil then
  begin
    OutProperties.Add('Action');      // ����������ֵ�������
    OutProperties.Add('Anchors');
    OutProperties.Add('Cancel');
    OutProperties.Add('Cursor');
    OutProperties.Add('DragMode');
    OutProperties.Add('Default');
    OutProperties.Add('DefaultDrawing');
    OutProperties.Add('Enabled');
    OutProperties.Add('GroupIndex');
    OutProperties.Add('HelpContext');
    OutProperties.Add('Hint');
    OutProperties.Add('ImageIndex');
    OutProperties.Add('Images');
    OutProperties.Add('ItemHeight');
    OutProperties.Add('ItemIndex');
    OutProperties.Add('Items.Strings');
    OutProperties.Add('Lines.Strings');
    OutProperties.Add('ModalResult');
    OutProperties.Add('ParentShowHint');
    OutProperties.Add('PopupMenu');
    OutProperties.Add('ReadOnly');
    OutProperties.Add('RowCount');
    OutProperties.Add('ShowHint');
    OutProperties.Add('ShortCut');
    OutProperties.Add('TabStop');
    OutProperties.Add('TabOrder');
    OutProperties.Add('Tag');
    OutProperties.Add('Text');
    OutProperties.Add('Visible');

    OutProperties.Add('ActivePage');   // ������Ҫ��������ֵ�����
    OutProperties.Add('ButtonHeight'); // ToolBar �� ButtonHeight/ButtonWidth
    OutProperties.Add('ButtonWidth');  // Ҫ�ĳ� TGridLayout �� ItemHeight/ItemWidth
    OutProperties.Add('Checked');      // TRadioButton/TCheckBox �� IsChecked
    OutProperties.Add('DefaultRowHeight'); // StringGrid �ĳ� RowHeight
    OutProperties.Add('PageIndex');
    OutProperties.Add('ScrollBars');   // ����������ֵ�����
    OutProperties.Add('TabPosition');  // ����������ĵ�����ֵҪ���
  end;
end;

class procedure TCnGeneralConverter.ProcessProperties(const PropertyName,
  TheClassName, PropertyValue: string; InProperties, OutProperties: TStrings;
  Tab: Integer);
var
  NewPropName: string;
begin
  // FMX �� TComboBox �� Text ���Բ����ڣ�����
  if (TheClassName = 'TComboBox') and (PropertyName = 'Text') then
    Exit;

  if PropertyName = 'ActivePage' then
    NewPropName := 'ActiveTab'
  else if PropertyName = 'PageIndex' then
    NewPropName := 'Index'
  else if (PropertyName = 'Checked') and ((TheClassName = 'TRadioButton') or
    (TheClassName = 'TCheckBox')) then
    NewPropName := 'IsChecked'
  else if (PropertyName = 'DefaultRowHeight') and (TheClassName = 'TStringGrid') then
    NewPropName := 'RowHeight'
  else if (PropertyName = 'ButtonWidth') and (TheClassName = 'TToolBar') then
  begin
    NewPropName := 'ItemWidth'
  end
  else if (PropertyName = 'ButtonHeight') and (TheClassName = 'TToolBar') then
  begin
    NewPropName := 'ItemHeight'
  end
  else if PropertyName = 'ScrollBars' then
  begin
    if PropertyValue = 'ssNone' then
      OutProperties.Add('ShowScrollBars = False')
    else
      OutProperties.Add('ShowScrollBars = True');
    Exit;    // ����ֵ���ˣ�д����˳���������д PropertyValue ��
  end
  else if PropertyName = 'TabPosition' then
  begin
    OutProperties.Add('TabPosition = ' + CnConvertEnumValue(PropertyValue));
    Exit;    // ����ֵ���ˣ�д����˳���������д PropertyValue ��
  end
  else if (PropertyName = 'MinValue') and (TheClassName = 'TSpinEdit') then
    NewPropName := 'Min'
  else if (PropertyName = 'MaxValue') and (TheClassName = 'TSpinEdit') then
    NewPropName := 'Max'
  else
    NewPropName := PropertyName;

  OutProperties.Add(Format('%s = %s', [NewPropName, PropertyValue]));
end;

{ TCnTreeViewConverter }

class procedure TCnTreeViewConverter.GetComponents(OutVclComponents: TStrings);
begin
  if OutVclComponents <> nil then
    OutVclComponents.Add('TTreeView');
end;

class procedure TCnTreeViewConverter.LoadTreeLeafFromStream(Root, Leaf: TCnDfmLeaf;
  Stream: TStream);
var
  I, Size: Integer;
  ALeaf: TCnDfmLeaf;
  Info: TNodeInfo;
begin
  Stream.ReadBuffer(Size, SizeOf(Size));
  Stream.ReadBuffer(Info, Size);

  // �� Info �������� Leaf �� Properties ��
  Leaf.ElementKind := dkObject;
  Leaf.ElementClass := 'TTreeViewItem';
  Leaf.Text := 'TTreeViewItem' + IntToStr(Leaf.Tree.GetSameClassIndex(Leaf) + 1);
  Leaf.Properties.Add('Text = ' + ConvertWideStringToDfmString(Info.Text));
  Leaf.Properties.Add('ImageIndex = ' + IntToStr(Info.ImageIndex));

  // �ݹ���������ӽڵ�
  for I := 0 to Info.Count - 1 do
  begin
    ALeaf := Leaf.Tree.AddChild(Leaf) as TCnDfmLeaf;
    LoadTreeLeafFromStream(Root, ALeaf, Stream);
  end;
end;

class procedure TCnTreeViewConverter.ProcessComponents(SourceLeaf,
  DestLeaf: TCnDfmLeaf; Tab: Integer);
var
  I, Count: Integer;
  Stream: TStream;
  Leaf: TCnDfmLeaf;
begin
  // ���� SourceLeaf �е� Items.Data ���������ݣ�����ת�����ӿؼ���ӵ���Ӧ DestLeaf ��
  I := IndexOfHead('Items.Data = ', SourceLeaf.Properties);
  if I >= 0 then
  begin
    if SourceLeaf.Properties.Objects[I] <> nil then
    begin
      // �� Stream �ж���ڵ���Ϣ
      Stream := TStream(SourceLeaf.Properties.Objects[I]);
      Stream.Position := 0;
      Stream.ReadBuffer(Count, SizeOf(Count));
      for I := 0 to Count - 1 do
      begin
        // �� DestLeaf ���һ���ӽڵ㣬��������ӽڵ������
        Leaf := DestLeaf.Tree.AddChild(DestLeaf) as TCnDfmLeaf;
        LoadTreeLeafFromStream(DestLeaf, Leaf, Stream);
      end;
    end;
  end;
end;

{ TCnImageConverter }

function LoadGraphicFromDfmBinStream(Stream: TStream): Vcl.Graphics.TGraphic;
var
  ClzName: ShortString;
  Clz: Vcl.Graphics.TGraphicClass;
begin
  Result := nil;
  if (Stream = nil) or (Stream.Size <= 0) then
    Exit;

  Stream.Read(ClzName[0], 1);
  Stream.Read(ClzName[1], Ord(ClzName[0]));

  // ע������Ҫ������ǰ׺�������� 'TBitmap' �ҵ� Vcl.Graphics ��� TBitmap
  // ��˵ñ�֤ uses ��Ҫ�� FMX.Graphics���� Vcl.Graphics
  Clz := Vcl.Graphics.TGraphicClass(FindClass(ClzName));
  if Clz <> nil then
  begin
    Result := Vcl.Graphics.TGraphic(Clz.NewInstance);
    Result.Create; // �ƺ��������������� TBitmap ���⴦��
    TVclGraphicAccess(Result).ReadData(Stream);
  end;
end;

class procedure TCnImageConverter.GetComponents(OutVclComponents: TStrings);
begin
  if OutVclComponents <> nil then
    OutVclComponents.Add('TImage');
end;

class procedure TCnImageConverter.ProcessComponents(SourceLeaf,
  DestLeaf: TCnDfmLeaf; Tab: Integer);
var
  I: Integer;
  Stream: TStream;
  TmpStream: TMemoryStream;
  AGraphic: Vcl.Graphics.TGraphic;
  FmxBitmap: FMX.Graphics.TBitmap;
begin
  I := IndexOfHead('Picture.Data = ', SourceLeaf.Properties);
  if I >= 0 then
  begin
    if SourceLeaf.Properties.Objects[I] <> nil then
    begin
      // �� Stream �ж��� Bitmap ��Ϣ
      Stream := TStream(SourceLeaf.Properties.Objects[I]);
      Stream.Position := 0;

      AGraphic := nil;
      FmxBitmap := nil;
      TmpStream := nil;

      try
        AGraphic := LoadGraphicFromDfmBinStream(Stream);

        if (AGraphic <> nil) and not AGraphic.Empty then
        begin
          TmpStream := TMemoryStream.Create;
          AGraphic.SaveToStream(TmpStream);

          FmxBitmap := FMX.Graphics.TBitmap.Create;
          FmxBitmap.LoadFromStream(TmpStream);
          TmpStream.Clear;
          FmxBitmap.SaveToStream(TmpStream);

          // TmpStream ���Ѿ��� PNG �ĸ�ʽ�ˣ�д�� Bitmap.PNG ��Ϣ
          TmpStream.Position := 0;
          DestLeaf.Properties.Add('Bitmap.PNG = {' +
            ConvertStreamToHexDfmString(TmpStream) + '}');
        end;
      finally
        AGraphic.Free;
        FmxBitmap.Free;
        TmpStream.Free;
      end;
    end;
  end;
end;

{ TCnGridConverter }

class procedure TCnGridConverter.GetComponents(OutVclComponents: TStrings);
begin
  if OutVclComponents <> nil then
    OutVclComponents.Add('TStringGrid');
end;

class procedure TCnGridConverter.ProcessComponents(SourceLeaf,
  DestLeaf: TCnDfmLeaf; Tab: Integer);
var
  I, Count: Integer;
  OptionStr, WidthStr, Value: string;
  Options: TStringList;
  Leaf: TCnDfmLeaf;
begin
  I := IndexOfHead('Options = ', SourceLeaf.Properties);
  if I >= 0 then
    OptionStr := SourceLeaf.Properties[I]
  else // Ĭ������
    OptionStr := '[goFixedVertLine, goFixedHorzLine, goVertLine, goHorzLine, goRangeSelect]';

  Delete(OptionStr, 1, Length('Options = '));
  Options := TStringList.Create;
  try
    ConvertSetStringToElements(OptionStr, Options);

    // ת������Ԫ�أ������ڶ�Ӧ��ϵ����ɾ��
    for I := Options.Count - 1 downto 0 do
    begin
      OptionStr := CnConvertEnumValueIfExists(Options[I]);
      if OptionStr <> '' then
        Options[I] := OptionStr
      else
        Options.Delete(I);
    end;

    OptionStr := ConvertSetElementsToString(Options);
    DestLeaf.Properties.Add('Options = ' + OptionStr);
  finally
    Options.Free;
  end;

  I := IndexOfHead('ColCount = ', SourceLeaf.Properties);
  if I >= 0 then
  begin
    Value := SourceLeaf.Properties[I];
    Delete(Value, 1, Length('ColCount = '));

    Count := StrToIntDef(Value ,0);
    if Count > 0 then
    begin
      I := IndexOfHead('DefaultColWidth = ', SourceLeaf.Properties);
      if I >= 0 then
      begin
        Value := SourceLeaf.Properties[I];
        Delete(Value, 1, Length('DefaultColWidth = '));

        WidthStr := GetFloatStringFromInteger(StrToIntDef(Value ,64));
      end
      else
        WidthStr := '';

      for I := 1 to Count do
      begin
        // �� DestLeaf ���һ���ӽڵ㣬û��������
        Leaf := DestLeaf.Tree.AddChild(DestLeaf) as TCnDfmLeaf;
        Leaf.ElementClass := 'TStringColumn';
        Leaf.ElementKind := dkObject;

        // ��ָ����Ⱦ�д��
        if WidthStr <> '' then
          Leaf.Properties.Add('Size.Width = ' + WidthStr);
        Leaf.Text := 'StringColumn' + IntToStr(Leaf.Tree.GetSameClassIndex(Leaf) + 1);
      end;
    end;
  end;
end;

{ TCnImageListConverter }

class procedure TCnImageListConverter.GetComponents(OutVclComponents: TStrings);
begin
  if OutVclComponents <> nil then
    OutVclComponents.Add('TImageList');
end;

class procedure TCnImageListConverter.ProcessComponents(SourceLeaf,
  DestLeaf: TCnDfmLeaf; Tab: Integer);
var
  I: Integer;
  Stream: TStream;
  ImageList: TImageList;
  ImgStr: string;
  TmpStream: TMemoryStream;
  VclBitmap: Vcl.Graphics.TBitmap;
  FmxBitmap: FMX.Graphics.TBitmap;

  procedure TrySetPropertyToImageList(const PropName: string);
  var
    Idx: Integer;
  begin
    Idx := IndexOfHead(PropName + ' = ', SourceLeaf.Properties);
    if Idx >= 0 then
    begin
      Idx := GetIntPropertyValue(SourceLeaf.Properties[Idx], PropName);
      SetOrdProp(ImageList, PropName, Idx);
    end;
  end;

begin
  I := IndexOfHead('Bitmap = ', SourceLeaf.Properties);
  if I >= 0 then
  begin
    if SourceLeaf.Properties.Objects[I] <> nil then
    begin
      // �� Stream �ж��� Bitmap ��Ϣ
      Stream := TStream(SourceLeaf.Properties.Objects[I]);
      Stream.Position := 0;

      ImageList := TImageList.Create(nil);
      try
        TrySetPropertyToImageList('Height');
        TrySetPropertyToImageList('Width');
        TrySetPropertyToImageList('AllocBy');

        TVclImageListAccess(ImageList).ReadData(Stream);

      // TODO: �� ImageList �а������⻭
      // д   Source = <
      // ѭ��д item
      //          MultiResBitmap.LoadSize = 0
      //          MultiResBitmap = <
      //            item
      //              Scale = 1.000000000000000000
      //              Width = 16
      //              Height = 16
      //              PNG = {
      //              }
      //              FileName = '\\VBOXSVR\Downloads\delphi5_setup\delphi.ico'
      //            end>
      //          Name = 'delphi'
      //        end
      // ����ѭ��д >

        if ImageList.Count > 0 then
        begin
          // д Source
          ImgStr := 'Source = <' + #13#10;
          for I := 0 to ImageList.Count - 1 do
          begin
            ImgStr := ImgStr + StringOfChar(' ', 2) + 'item' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 4) + 'MultiResBitmap.LoadSize = 0' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 4) + 'MultiResBitmap = < ' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 6) + 'item' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 8) + 'Scale = 1.000000000000000000' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 8) + 'Width = ' + IntToStr(ImageList.Width) + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 8) + 'Height = ' + IntToStr(ImageList.Height) + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 8) + 'PNG = {';

            VclBitmap := nil;
            TmpStream := nil;
            FmxBitmap := nil;
            try
              VclBitmap := Vcl.Graphics.TBitmap.Create;
              VclBitmap.Width := ImageList.Width;
              VclBitmap.Height := ImageList.Height;
              ImageList.Draw(VclBitmap.Canvas, 0, 0, I, True);

              TmpStream := TMemoryStream.Create;
              VclBitmap.SaveToStream(TmpStream);
              TmpStream.Position := 0;
              FmxBitmap := Fmx.Graphics.TBitmap.Create;
              FmxBitmap.LoadFromStream(TmpStream);
              TmpStream.Clear;
              FmxBitmap.SaveToStream(TmpStream);

              TmpStream.Position := 0;
              ImgStr := ImgStr + ConvertStreamToHexDfmString(TmpStream, 10) + '}' + #13#10;
            finally
              VclBitmap.Free;
              FmxBitmap.Free;
              TmpStream.Free;
            end;

            // ImgStr := ImgStr + StringOfChar(' ', 8) + 'FileName = ''''' + #13#10; û���ļ���
            ImgStr := ImgStr + StringOfChar(' ', 6) + 'end>' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 4) + 'Name = ''Item ' + IntToStr(I) + ''''#13#10;

            // ��������һͼ����β
            if I = ImageList.Count - 1 then
              ImgStr := ImgStr + StringOfChar(' ', 2) + 'end>' + #13#10
            else
              ImgStr := ImgStr + StringOfChar(' ', 2) + 'end' + #13#10;
          end;

          // д Destination
          ImgStr := ImgStr + 'Destination = <' + #13#10;
          for I := 0 to ImageList.Count - 1 do
          begin
            ImgStr := ImgStr + StringOfChar(' ', 2) + 'item' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 4) + 'Layers = <' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 6) + 'item' + #13#10;
            ImgStr := ImgStr + StringOfChar(' ', 8) + 'Name = ''Item ' + IntToStr(I) + ''''#13#10;
            ImgStr := ImgStr + StringOfChar(' ', 6) + 'end>' + #13#10;

            // ��������һͼ����β
            if I = ImageList.Count - 1 then
              ImgStr := ImgStr + StringOfChar(' ', 2) + 'end>'
            else
              ImgStr := ImgStr + StringOfChar(' ', 2) + 'end' + #13#10;
          end;

          DestLeaf.Properties.Add(ImgStr);
        end;
      finally
        ImageList.Free;
      end;
    end;
  end;
end;

{ TCnToolBarConverter }

class procedure TCnToolBarConverter.GetComponents(OutVclComponents: TStrings);
begin
  if OutVclComponents <> nil then
    OutVclComponents.Add('TToolBar');
end;

class procedure TCnToolBarConverter.ProcessComponents(SourceLeaf,
  DestLeaf: TCnDfmLeaf; Tab: Integer);
var
  I: Integer;
  S: string;
begin
  I := IndexOfHead('Images = ', DestLeaf.Properties);
  if I >= 0 then
  begin
    S := DestLeaf.Properties[I];
    Delete(S, 1, Length('Images = '));
    DestLeaf.Properties.Delete(I);

    for I := 0 to DestLeaf.Count - 1 do
    begin
      if IndexOfHead('Images = ', DestLeaf.Items[I].Properties) < 0 then
        DestLeaf.Items[I].Properties.Add('Images = ' + S);
    end;
  end;
end;

function MergeSource(const SourceFile, FormClass, NewFileName: string;
  UsesList, FormDecl: TStrings): string;
var
  SrcStream, ResStream: TMemoryStream;
  SrcStr, S, ANewFileName: string;
  SrcList: TStringList;
  C: Char;
  P: PByteArray;
  L, L1: Integer;
  Lex: TCnPasWideLex;
  UnitNamePos, UnitNameLen, UsesPos, UsesEndPos, FormTokenPos, PrivatePos: Integer;
  InImpl, InUses, UnitGot, UsesGot, TypeGot, InForm: Boolean;
  FormGot, FormGot1, FormGot2, FormGot3: Boolean;
begin
  // 1����Դ Pas ͷ���� implementation ���ֵĵ�һ�� uses �ؼ��֣�����
  // 2�������� uses �е����ݣ��� Units �ϲ���ȥ���滻ԭʼ��
  // 3���ٵ� type ����ԭʼ Form ������TFormXXX = class(TForm) ��Ҫʶ�𵽣��� TFormXXX ����һ�� private/protected/public ֮ǰ���滻
  // 4�����Ƶ��ļ�β����Ѱ�� implementation ��� {$R *.dfm}���滻�� {$R *.fmx}

  SrcList := nil;
  SrcStream := nil;
  ResStream := nil;
  Lex := nil;

  try
    SrcList := TStringList.Create;
    SrcList.LoadFromFile(SourceFile);

    SrcStr := SrcList.Text;
    SrcStream := TMemoryStream.Create;

    SrcStream.Write(SrcStr[1], Length(SrcStr) * SizeOf(Char));
    C := #0;
    SrcStream.Write(C, SizeOf(Char));

    Lex := TCnPasWideLex.Create(True);
    Lex.Origin := PWideChar(SrcStream.Memory);

    InImpl := False;
    UnitGot := False;
    UsesGot := False;
    TypeGot := False;
    InForm := False;
    FormGot1 := False;
    FormGot2 := False;
    FormGot3 := False;
    FormGot := False;

    UnitNamePos := 0;
    UnitNameLen := 0;
    UsesPos := 0;
    UsesEndPos := 0;
    FormTokenPos := 0;
    PrivatePos := 0;

    while Lex.TokenID <> tkNull do
    begin
      case Lex.TokenID of
        tkUnit:
          begin
            UnitGot := True;
          end;
        tkUses:
          begin
            if not UsesGot and not InImpl then
            begin
              UsesGot := True;
              InUses := True;
              // ��¼ uses ��λ��
              UsesPos := Lex.TokenPos;
            end;
          end;
        tkSemiColon:
          begin
            if InUses then
            begin
              InUses := False;
              // ��¼ uses ��β�ķֺŵ�λ��
              UsesEndPos := Lex.TokenPos;
            end;
          end;
        tkType:
          begin
            if not InImpl then
              TypeGot := True;
          end;
        tkImplementation:
          begin
            InImpl := True;
          end;
        tkPrivate, tkProtected, tkPublic:
          begin
            if InForm then
            begin
              // ��¼��ǰ private ��λ��
              PrivatePos := Lex.TokenPos;
              InForm := False;
            end;
          end;
        tkIdentifier:
          begin
            if UnitGot then
            begin
              // ��ǰ��ʶ���ǵ�Ԫ��
              UnitNamePos := Lex.TokenPos;
              UnitNameLen := Lex.TokenLength;
              UnitGot := False;
            end;

            if TypeGot and not FormGot and (Lex.Token = FormClass) then
            begin
              FormGot1 := True;
              FormGot2 := False;
              FormGot3 := False;
              InForm := False;

              FormTokenPos := Lex.TokenPos;
            end;
          end;
        tkEqual:
          begin
            if FormGot1 then
            begin
              FormGot2 := True;
              FormGot1 := False;
              FormGot3 := False;
            end;
          end;
        tkClass:
          begin
            if FormGot2 then
            begin
              FormGot3 := True;
              FormGot1 := False;
              FormGot2 := False;
            end;
          end;
        tkRoundOpen:
          begin
            if FormGot3 then
            begin
              InForm := True;  // ������ȷ���������ˣ�֮ǰ��¼�� FormTokenPos ��Ч
              FormGot := True;

              FormGot1 := False;
              FormGot2 := False;
              FormGot3 := False;
            end;
          end;
      end;

      if not (Lex.TokenID in [tkIdentifier, tkClass, tkEqual, tkRoundOpen,
        tkCompDirect, tkAnsiComment, tkBorComment]) then
      begin
        FormGot1 := False;
        FormGot2 := False;
        FormGot3 := False;
      end;

      Lex.NextNoJunk;
    end;

    // ��ͷд�� uses �������Ļس�
    // д�µ� uses �б�
    // �� usesEndPos д�� FormTokenPos
    // д�� Form ������������¼��б�
    // д privatePos ��β
    // ����滻 {$R *.dfm}
    if (UsesPos = 0) or (UsesEndPos = 0) or (FormTokenPos = 0) or (PrivatePos = 0) then
      Exit;

    P := PByteArray(SrcStream.Memory);
    ResStream := TMemoryStream.Create;

    L := UnitNamePos;
    SetLength(S, L);
    Move(P^[0], S[1], UnitNamePos * SizeOf(Char));
    ResStream.Write(S[1], Length(S) * SizeOf(Char)); // ��ͷд�� unit
    if NewFileName = '' then
      ANewFileName := UNIT_NAME_TAG
    else
      ANewFileName := ChangeFileExt(ExtractFileName(NewFileName), '');

    ResStream.Write(UNIT_NAME_TAG[1], Length(UNIT_NAME_TAG) * SizeOf(Char)); // д��Ԫ����ʶ

    L := UsesPos - (UnitNamePos + UnitNameLen) + Length('uses');
    SetLength(S, L);
    Move(P^[(UnitNamePos + UnitNameLen) * SizeOf(Char)], S[1], L * SizeOf(Char));
    ResStream.Write(S[1], Length(S) * SizeOf(Char)); // �� UnitName βд�� uses

    L := UsesPos + Length('uses'); // �ָ� L
    L1 := UsesEndPos - L;
    SetLength(S, L1);
    Move(P^[L * SizeOf(Char)], S[1], L1 * SizeOf(Char));
    S := ReplaceVclUsesToFmx(S, UsesList);
    ResStream.Write(S[1], Length(S) * SizeOf(Char)); // д uses ��Ԫ�б�

    L := FormTokenPos - UsesEndPos;
    SetLength(S, L);
    Move(P^[UsesEndPos * SizeOf(Char)], S[1], L * SizeOf(Char));
    ResStream.Write(S[1], Length(S) * SizeOf(Char)); // д uses ��Ԫ�б�� Form ��������

    S := FormDecl.Text;
    ResStream.Write(S[1], Length(S) * SizeOf(Char)); // д Form ����

    C := ' ';
    ResStream.Write(C, SizeOf(Char));
    ResStream.Write(C, SizeOf(Char));   // private ǰ�����ո�����

    L := (SrcStream.Size div SizeOf(Char)) - PrivatePos;
    SetLength(S, L);
    Move(P^[PrivatePos * SizeOf(Char)], S[1], L * SizeOf(Char));
    ResStream.Write(S[1], Length(S) * SizeOf(Char)); // д��β

    ResStream.Size := ResStream.Size - SizeOf(Char); // ȥ��ĩβ�� #0

    // �滻 OutStream �е� {$R *.dfm}
    SetLength(Result, ResStream.Size div SizeOf(Char));
    Move(ResStream.Memory^, Result[1], ResStream.Size);
    Result := StringReplace(Result, '{$R *.dfm}', '{$R *.fmx}', [rfIgnoreCase]);
  finally
    SetLength(S, 0);
    Lex.Free;
    SrcStream.Free;
    ResStream.Free;
    SrcList.Free;
  end;
end;

function CnVclToFmxConvert(const InDfmFile: string; var OutFmx, OutPas: string;
  const NewFileName: string): Boolean;
var
  ATree, ACloneTree: TCnDfmTree;
  I, L: Integer;
  OutClass, OS, NS: string;
  List, FormDecl, EventIntf, EventImpl, Units, SinglePropMap: TStringList;
begin
  Result := False;
  if not FileExists(InDfmFile) then
    Exit;

  ATree := nil;
  ACloneTree := nil;
  List := nil;
  FormDecl := nil;
  EventIntf := nil;
  EventImpl := nil;
  SinglePropMap := nil;
  Units := nil;

  try
    ATree := TCnDfmTree.Create;
    if LoadDfmFileToTree(InDfmFile, ATree) then
    begin
      ACloneTree := TCnDfmTree.Create;
      ACloneTree.Assign(ATree);
    end
    else
      Exit; // dfm ���Ϸ�

    if (ATree.Count <> ACloneTree.Count) or (ATree.Count < 2) then
      Exit;

    FormDecl := TStringList.Create;
    EventIntf := TStringList.Create;
    EventImpl := TStringList.Create;
    SinglePropMap := TStringList.Create;
    Units := TStringList.Create;

    Units.Sorted := True;
    Units.Duplicates := dupIgnore;

    CnConvertTreeFromVclToFmx(ATree, ACloneTree, EventIntf, EventImpl, Units, SinglePropMap);

    // ��� ACloneTree �����⴦����� TabControl �� Sizes ����ֵ
    HandleTreeSpecial(ACloneTree);

    OutClass := '  ' + Units[0];
    for I := 1 to Units.Count - 1 do
      OutClass := OutClass + ', ' + Units[I];
    OutClass := OutClass + ';';

    with FormDecl do
    begin
      Add(ACloneTree.Items[1].ElementClass + ' = class(TForm)');
      for I := 2 to ACloneTree.Count - 1 do
        if ACloneTree.Items[I].ElementClass <> '' then
          Add('    ' + ACloneTree.Items[I].Text + ': '
            + ACloneTree.Items[I].ElementClass + ';');
      AddStrings(EventIntf);
    end;

    List := TStringList.Create;
    SaveTreeToStrings(List, ACloneTree);
    OutFmx := List.Text;

    if not FileExists(ChangeFileExt(InDfmFile, '.pas')) then
    begin
      Result := True;
      Exit;
    end;

    // ��ʱ FormDecl �� FMX �� published ���ֵ�������
    // ��Ҫ��ԭʼ�ļ��� private �������ƴ������
    OutPas := MergeSource(ChangeFileExt(InDfmFile, '.pas'),
      ACloneTree.Items[1].ElementClass, NewFileName, Units, FormDecl);

    // �滻Դ���еı�ʶ��
    for I := 0 to SinglePropMap.Count - 1 do
    begin
      L := Pos('=', SinglePropMap[I]);
      if L > 0 then
      begin
        OS := Copy(SinglePropMap[I], 1, L - 1);
        NS := Copy(SinglePropMap[I], L + 1, MaxInt);
        if (OS <> '') and (NS <> '') then
          OutPas := CnStringReplace(OutPas, OS, NS, [crfReplaceAll,
            crfIgnoreCase, crfWholeWord]);
      end;
    end;

    Result := True;
  finally
    FormDecl.Free;
    EventIntf.Free;
    EventImpl.Free;
    SinglePropMap.Free;
    Units.Free;
    List.Free;
    ACloneTree.Free;
    ATree.Free;
  end;
end;

function CnVclToFmxReplaceUnitName(const PasName, InPas: string): string;
var
  UName: string;
begin
  UName := ChangeFileExt(ExtractFileName(PasName), '');
  Result := StringReplace(InPas, UNIT_NAME_TAG, UName, []);
end;

// ���ļ����ݱ������ļ���fmx �ļ�Ĭ�� Ansi ���룬�ڲ��� Unicode ת�壬Դ���ļ�Ĭ�� Utf8 ����
function CnVclToFmxSaveContent(const FileName, InFile: string; SourceUtf8: Boolean): Boolean;
var
  M: TBytesStream;
  T: TBytes;
begin
  Result := False;
  if SourceUtf8 then
    T := TEncoding.UTF8.GetBytes(InFile)
  else
    T := TEncoding.ANSI.GetBytes(InFile);

  M := TBytesStream.Create(T);
  try
    M.SaveToFile(FileName);
    Result := True;
  finally
    M.Free;
  end;
end;

initialization
  RegisterCnPropertyConverter(TCnPositionConverter);
  RegisterCnPropertyConverter(TCnSizeConverter);
  RegisterCnPropertyConverter(TCnCaptionConverter);
  RegisterCnPropertyConverter(TCnFontConverter);
  RegisterCnPropertyConverter(TCnTouchConverter);
  RegisterCnPropertyConverter(TCnGeneralConverter);

  RegisterCnComponentConverter(TCnTreeViewConverter);
  RegisterCnComponentConverter(TCnImageConverter);
  RegisterCnComponentConverter(TCnGridConverter);
  RegisterCnComponentConverter(TCnImageListConverter);
  RegisterCnComponentConverter(TCnToolBarConverter);

  RegisterClasses([TIcon, TBitmap, TMetafile, TWICImage, TJpegImage, TGifImage, TPngImage]);

end.
