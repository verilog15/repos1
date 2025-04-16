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

unit CnCorPropRulesFrm;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ������޸Ĺ���༭��Ԫ
* ��Ԫ���ߣ���ʡ(hubdog) hubdog@263.net
*           CnPack ������ master@cnpack.org
* ��    ע�������޸�ר�����õ�Ԫ
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin2000 + Delphi 5
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2022.10.06 V1.2 by LiuXiao
*               ����ʹ�� CnSearchComboBox �滻���� ComboBox
*           2004.05.15 V1.1 by LiuXiao
*               �޸� PropDef ���õ��ظ��ͷŵ��³��������
*           2003.05.17 V1.0 by LiuXiao
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

{$IFDEF CNWIZARDS_CNCORPROPWIZARD}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, TypInfo, CnCommon, CnWizConsts, CnWizUtils, CnCorPropWizard,
  CnWizMultiLang, CnSearchCombo;

type
  TCnCorPropRuleForm = class(TCnTranslateForm)
    btnOK: TButton;
    btnCancel: TButton;
    btnHelp: TButton;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    cbbComponent: TComboBox;
    cbbProperty: TComboBox;
    cbbCondition: TComboBox;
    cbbValue: TComboBox;
    cbbAction: TComboBox;
    cbbDestValue: TComboBox;
    chkActive: TCheckBox;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
    procedure FormShow(Sender: TObject);
    procedure btnHelpClick(Sender: TObject);
    procedure cbbComponentChange(Sender: TObject);
  private
    FPropDef: TCnPropDef;
    FcbbComponent: TCnSearchComboBox;
    FcbbProperty: TCnSearchComboBox;
    procedure SetPropDef(const Value: TCnPropDef);
    function GetPropDef: TCnPropDef;
  protected
    function GetHelpTopic: string; override;
  public
    procedure ClearAll;
    procedure AddUniqueToCombo(Combo: TComboBox);
    property PropDef: TCnPropDef read GetPropDef write SetPropDef;
    {������ã��Լ�������}
  end;

var
  CorPropRuleForm: TCnCorPropRuleForm = nil;

{$ENDIF CNWIZARDS_CNCORPROPWIZARD}

implementation

{$IFDEF CNWIZARDS_CNCORPROPWIZARD}

uses
  CnCorPropCfgFrm, CnCorPropFrm, CnWizIdeUtils, CnWizOptions;

{$R *.DFM}

procedure TCnCorPropRuleForm.FormCreate(Sender: TObject);
var
  I: Integer;
begin
  FPropDef := TCnPropDef.Create(nil);
  cbbCondition.Items.Clear;
  cbbAction.Items.Clear;
  for I := Ord(Low(CompareStr)) to Ord(High(CompareStr)) do
    cbbCondition.Items.Add(CompareStr[TCompareOper(I)]);

  ActionStr[paWarn] := SCnCorrectPropertyActionWarn;
  ActionStr[paCorrect] := SCnCorrectPropertyActionAutoCorrect;
  for I := Ord(Low(ActionStr)) to Ord(High(ActionStr)) do
    cbbAction.Items.Add(ActionStr[TPropAction(I)]);

{$IFDEF COMPILER6_UP}
  cbbComponent.AutoComplete := True;
  cbbProperty.AutoComplete := True;
  cbbValue.AutoComplete := True;
  cbbDestValue.AutoComplete := True;
{$ENDIF}

  cbbCondition.ItemIndex := 0;
  cbbAction.ItemIndex := 0;

  if WizOptions.UseSearchCombo then
  begin
    CloneSearchCombo(FcbbComponent, cbbComponent);
    CloneSearchCombo(FcbbProperty, cbbProperty);
  end;
end;

procedure TCnCorPropRuleForm.SetPropDef(const Value: TCnPropDef);
begin
  if not Assigned(Value) then
    Exit;

  with Value do
  begin
    if WizOptions.UseSearchCombo then
    begin
      FcbbComponent.SetTextWithoutChange(CompName);
      FcbbProperty.SetTextWithoutChange(PropName);
    end
    else
    begin
      cbbComponent.Text := CompName;
      cbbProperty.Text := PropName;
    end;

    cbbCondition.ItemIndex := Ord(Compare);
    cbbValue.Text := Value;
    cbbAction.ItemIndex := Ord(Action);
    cbbDestValue.Text := ToValue;
    chkActive.Checked := Active;
  end;
  FPropDef.Assign(Value);
end;

procedure TCnCorPropRuleForm.FormDestroy(Sender: TObject);
begin
  FreeAndNil(FPropDef);
end;

function TCnCorPropRuleForm.GetPropDef: TCnPropDef;
begin
  if FPropDef <> nil then with FPropDef do
  begin
    if WizOptions.UseSearchCombo then
    begin
      CompName := FcbbComponent.Text;
      PropName := FcbbProperty.Text;
    end
    else
    begin
      CompName := cbbComponent.Text;
      PropName := cbbProperty.Text;
    end;

    Compare := TCompareOper(cbbCondition.ItemIndex);
    Value := cbbValue.Text;
    Action := TPropAction(cbbAction.ItemIndex);
    ToValue := cbbDestValue.Text;
    Active := chkActive.Checked;
  end;
  Result := FPropDef;
end;

procedure TCnCorPropRuleForm.ClearAll;
begin
  if WizOptions.UseSearchCombo then
  begin
    FcbbComponent.Text := '';
    FcbbProperty.Text := '';
  end
  else
  begin
    cbbComponent.Text := '';
    cbbProperty.Text := '';
  end;

  cbbValue.Text := '';
  cbbDestValue.Text := '';
  chkActive.Checked := True;
end;

procedure TCnCorPropRuleForm.AddUniqueToCombo(Combo: TComboBox);
begin
  if (Combo <> nil) and
     (Combo.Style <> csDropDownList) and
     (Combo.Text <> '') and
     (Combo.Items.IndexOf(Combo.Text) < 0) then
  begin
    Combo.Items.Add(Combo.Text);
  end;
end;

procedure TCnCorPropRuleForm.FormCloseQuery(Sender: TObject;
  var CanClose: Boolean);
var
  AClass: TPersistentClass;
  AComponent: TComponent;
  CompText, PropText: string;
begin
  if ModalResult = mrOK then
  begin
    CanClose := True;
    if WizOptions.UseSearchCombo then
    begin
      CompText := FcbbComponent.Text;
      PropText := FcbbProperty.Text;
    end
    else
    begin
      CompText := cbbComponent.Text;
      PropText := cbbProperty.Text;
    end;

    AClass := GetClass(CompText);
    if AClass = nil then
    begin
      CanClose := QueryDlg(Format(SCnCorrectPropertyErrClassFmt,
        [CompText]));
    end
    else
    begin
      if GetPropInfo(AClass, PropText) = nil then
      begin
        AComponent := nil;
        try
          AComponent := TComponent(AClass.NewInstance);
          try
            AComponent.Create(nil);
          except
            AComponent := nil;
            CanClose := QueryDlg(Format(SCnCorrectPropertyErrClassCreate,
              [CompText, PropText]));
          end;

          if (AComponent <> nil) and (GetPropInfoIncludeSub(AComponent, PropText) = nil) then
            CanClose := QueryDlg(Format(SCnCorrectPropertyErrPropFmt,
              [CompText, PropText]));
        except
          CanClose := QueryDlg(Format(SCnCorrectPropertyErrPropFmt,
            [CompText, PropText]));
        end;

        try
          AComponent.Free;
        except
          ;
        end;
      end;
    end;
  end
  else
  begin
    CanClose := True;
  end;

  if CanClose then
  begin
    AddUniqueToCombo(cbbProperty);
    AddUniqueToCombo(cbbValue);
    AddUniqueToCombo(cbbDestValue);
  end;
end;

procedure TCnCorPropRuleForm.FormShow(Sender: TObject);
var
  I: Integer;
begin
  if WizOptions.UseSearchCombo then
  begin
    with FcbbComponent do
    begin
      GetInstalledComponents(nil, Items);
      for I := 0 to CnNoIconList.Count - 1 do
        Items.Add(CnNoIconList[I]);
      OnSelect(FcbbComponent);
      SetFocus;
    end;
  end
  else
  begin
    with cbbComponent do
    begin
      GetInstalledComponents(nil, Items);
      for I := 0 to CnNoIconList.Count - 1 do
        Items.Add(CnNoIconList[I]);
      OnChange(cbbComponent);
      SetFocus;
    end;
  end;
end;

procedure TCnCorPropRuleForm.cbbComponentChange(Sender: TObject);
var
  AClass: TClass;
begin
  try
    if WizOptions.UseSearchCombo then
    begin
      FcbbProperty.Items.Clear;

      AClass := FindClass(FcbbComponent.Text);
      if AClass <> nil then
        GetAllPropNamesFromClass(AClass, FcbbProperty.Items); // ������ҵ��࣬�������б��ҳ���
    end
    else
    begin
      cbbProperty.Items.Clear;

      AClass := FindClass(cbbComponent.Text);
      if AClass <> nil then
        GetAllPropNamesFromClass(AClass, cbbProperty.Items); // ������ҵ��࣬�������б��ҳ���
    end;
  except
    ;
  end;
end;

procedure TCnCorPropRuleForm.btnHelpClick(Sender: TObject);
begin
  ShowFormHelp;
end;

function TCnCorPropRuleForm.GetHelpTopic: string;
begin
  Result := 'CnCorrectProperty';
end;

{$ENDIF CNWIZARDS_CNCORPROPWIZARD}
end.

