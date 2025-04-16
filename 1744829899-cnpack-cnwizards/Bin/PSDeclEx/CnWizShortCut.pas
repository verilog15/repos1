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

unit CnWizShortCut;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�IDE ��ݼ�����͹�����ʵ�ֵ�Ԫ
* ��Ԫ���ߣ��ܾ��� (zjy@cnpack.org)
* ����ƽ̨��PWin2000Pro + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2017.05.18 V1.0
*               ������Ԫ��ʵ�ֹ���
================================================================================
|</PRE>}

interface

uses
  Windows, Messages, Classes, SysUtils, Menus, ExtCtrls, ToolsAPI,
  CnWizConsts, CnCommon;

type
//==============================================================================
// IDE ��ݼ�������
//==============================================================================

{ TCnWizShortCut }

  TCnWizShortCutMgr = class;

  TCnWizShortCut = class(TObject)
  {* IDE ��ݼ������࣬�� CnWizards ��ʹ�ã������˿�ݼ������Ժ͹��ܡ�
     ÿһ���������Ŀ�ݼ�ʵ�������Զ��ڴ���ʱ��ע�����ؿ�ݼ��������ͷ�ʱ
     ���б��档�벻Ҫֱ�Ӵ������ͷŸ����ʵ������Ӧ��ʹ�ÿ�ݼ�������
     WizShortCutMgr �� Add �� Delete ������ʵ�֡�}
  private
    FDefShortCut: TShortCut;
    FOwner: TCnWizShortCutMgr;
    FShortCut: TShortCut;
    FKeyProc: TNotifyEvent;
    FMenuName: string;
    FName: string;
    FTag: Integer;
    procedure SetKeyProc(const Value: TNotifyEvent);
    procedure SetShortCut(const Value: TShortCut);
    procedure SetMenuName(const Value: string);
    function ReadShortCut(const Name: string; DefShortCut: TShortCut): TShortCut;
    procedure WriteShortCut(const Name: string; AShortCut: TShortCut);
  protected
    procedure Changed; virtual;
  public
    constructor Create(AOwner: TCnWizShortCutMgr; const AName: string;
      AShortCut: TShortCut; AKeyProc: TNotifyEvent; const AMenuName: string;
      ATag: Integer = 0);
    {* �๹�������벻Ҫֱ�ӵ��ø÷���������ʵ������Ӧ���ÿ�ݼ�������
       WizShortCutMgr.Add ���������������� Delete ��ɾ����}
    destructor Destroy; override;
    {* �����������벻Ҫֱ���ͷŸ����ʵ������Ӧ���ÿ�ݼ�������
       WizShortCutMgr.Delete ��ɾ��һ�� IDE ��ݼ���}

    property Name: string read FName;
    {* ��ݼ������֣�ͬʱҲ�Ǳ�����ע����еļ�ֵ�������Ϊ�գ��ÿ�ݼ�����
       ������ע����С�}
    property ShortCut: TShortCut read FShortCut write SetShortCut;
    {* ��ݼ���ֵ��}
    property KeyProc: TNotifyEvent read FKeyProc write SetKeyProc;
    {* ��ݼ�֪ͨ�¼�������ݼ�������ʱ���ø��¼�}
    property MenuName: string read FMenuName write SetMenuName;
    {* ���ݼ������� IDE �˵��������}
    property Tag: Integer read FTag write FTag;
    {* ��ݼ���ǩ}
  end;

//==============================================================================
// IDE ��ݼ���������
//==============================================================================

{ TCnWizShortCutMgr }

  TCnWizShortCutMgr = class(TObject)
  {* IDE ��ݼ��������࣬����ά���� IDE �а󶨵Ŀ�ݼ����б�
     �벻Ҫֱ�Ӵ��������ʵ����Ӧʹ�� WizShortCutMgr ����õ�ǰ�Ĺ�����ʵ����}
  private
    FShortCuts: TList;
    FKeyBindingIndex: Integer;
    FUpdateCount: Integer;
    FUpdated: Boolean;
    FSaveMenus: TList;
    FSaveShortCuts: TList;
    FMenuTimer: TTimer;
    function GetShortCuts(Index: Integer): TCnWizShortCut;
    function GetCount: Integer;
    procedure InstallKeyBinding;
    procedure RemoveKeyBinding;
    procedure SaveMainMenuShortCuts;
    procedure RestoreMainMenuShortCuts;
    procedure DoRestoreMainMenuShortCuts(Sender: TObject);
  public
    constructor Create;
    {* �๹�������벻Ҫֱ�Ӵ��������ʵ����Ӧʹ�� WizShortCutMgr ����õ�ǰ
       �Ĺ�����ʵ����}
    destructor Destroy; override;
    {* ����������}
    function IndexOfShortCut(AWizShortCut: TCnWizShortCut): Integer;
    {* ���� IDE ��ݼ�������������ţ�����Ϊ��ݼ�������������ڷ���-1��}
    function IndexOfName(const AName: string): Integer; 
    {* ���ݿ�ݼ����Ʋ��������ţ���������ڷ���-1��}
    function Add(const AName: string; AShortCut: TShortCut; AKeyProc:
      TNotifyEvent; const AMenuName: string = ''; ATag: Integer = 0): TCnWizShortCut;
    {* ����һ����ݼ�����
     |<PRE>
       AName: string           - ��ݼ����ƣ����Ϊ�մ���ÿ�ݼ������浽ע�����
       AShortCut: TShortCut    - ��ݼ�Ĭ�ϼ�ֵ����� AName ��Ч��ʵ��ʹ�õļ�ֵ�Ǵ�ע����ж�ȡ��
       AKeyProc: TNotifyEvent  - ��ݼ�֪ͨ�¼�
       AMenuName: string       - ��ݼ���Ӧ�� IDE ���˵���������û�п���Ϊ��
       Result: Integer;        - ���������ӵĿ�ݼ������ţ����Ҫ����Ŀ�ݼ��Ѵ��ڷ���-1
     |</PRE>}
    procedure Delete(Index: Integer);
    {* ɾ��һ����ݼ����󣬲���Ϊ��ݼ�����������š�}
    procedure DeleteShortCut(var AWizShortCut: TCnWizShortCut); 
    {* ɾ��һ����ݼ����󣬲���Ϊ��ݼ����󣬵��óɹ����������Ϊ nil��}
    procedure Clear;
    {* ��տ�ݼ������б�}
    procedure BeginUpdate;
    {* ��ʼ���¿�ݼ������б��ڴ�֮��Կ�ݼ��б�ĸĶ��������Զ����а�ˢ�£�
       ֻ�е����½�����Ż��Զ����°󶨡�
       ʹ��ʱ������ EndUpdate ��ԡ�}
    procedure EndUpdate;
    {* �����Կ�ݼ������б�ĸ��£���������һ�ε��ã����Զ����°� IDE ��ݼ���
       ʹ��ʱ������ BeginUpdate ��ԡ�}
    function Updating: Boolean;
    {* ��ݼ������б����״̬���� BeginUpdate �� EndUpdate}
    procedure UpdateBinding;
    {* �����Ѱ󶨵Ŀ�ݼ������б�}

    property Count: Integer read GetCount;
    {* ��ݼ�����������}
    property ShortCuts[Index: Integer]: TCnWizShortCut read GetShortCuts;
    {* ��ݼ���������}
  end;

function WizShortCutMgr: TCnWizShortCutMgr;
{* ���ص�ǰ�� IDE ��ݼ�������ʵ���������Ҫʹ�ÿ�ݼ����������벻Ҫֱ�Ӵ���
   TCnWizShortCutMgr ��ʵ������Ӧ���øú��������ʡ�}

implementation

{ TCnWizShortCut }

procedure TCnWizShortCut.Changed;
begin

end;

constructor TCnWizShortCut.Create(AOwner: TCnWizShortCutMgr;
  const AName: string; AShortCut: TShortCut; AKeyProc: TNotifyEvent;
  const AMenuName: string; ATag: Integer);
begin

end;

destructor TCnWizShortCut.Destroy;
begin
  inherited;

end;

function TCnWizShortCut.ReadShortCut(const Name: string;
  DefShortCut: TShortCut): TShortCut;
begin

end;

procedure TCnWizShortCut.SetKeyProc(const Value: TNotifyEvent);
begin

end;

procedure TCnWizShortCut.SetMenuName(const Value: string);
begin

end;

procedure TCnWizShortCut.SetShortCut(const Value: TShortCut);
begin

end;

procedure TCnWizShortCut.WriteShortCut(const Name: string;
  AShortCut: TShortCut);
begin

end;

{ TCnWizShortCutMgr }

function TCnWizShortCutMgr.Add(const AName: string; AShortCut: TShortCut;
  AKeyProc: TNotifyEvent; const AMenuName: string;
  ATag: Integer): TCnWizShortCut;
begin

end;

procedure TCnWizShortCutMgr.BeginUpdate;
begin

end;

procedure TCnWizShortCutMgr.Clear;
begin

end;

constructor TCnWizShortCutMgr.Create;
begin

end;

procedure TCnWizShortCutMgr.Delete(Index: Integer);
begin

end;

procedure TCnWizShortCutMgr.DeleteShortCut(
  var AWizShortCut: TCnWizShortCut);
begin

end;

destructor TCnWizShortCutMgr.Destroy;
begin
  inherited;

end;

procedure TCnWizShortCutMgr.DoRestoreMainMenuShortCuts(Sender: TObject);
begin

end;

procedure TCnWizShortCutMgr.EndUpdate;
begin

end;

function TCnWizShortCutMgr.GetCount: Integer;
begin

end;

function TCnWizShortCutMgr.GetShortCuts(Index: Integer): TCnWizShortCut;
begin

end;

function TCnWizShortCutMgr.IndexOfName(const AName: string): Integer;
begin

end;

function TCnWizShortCutMgr.IndexOfShortCut(
  AWizShortCut: TCnWizShortCut): Integer;
begin

end;

procedure TCnWizShortCutMgr.InstallKeyBinding;
begin

end;

procedure TCnWizShortCutMgr.RemoveKeyBinding;
begin

end;

procedure TCnWizShortCutMgr.RestoreMainMenuShortCuts;
begin

end;

procedure TCnWizShortCutMgr.SaveMainMenuShortCuts;
begin

end;

procedure TCnWizShortCutMgr.UpdateBinding;
begin

end;

function TCnWizShortCutMgr.Updating: Boolean;
begin

end;

function WizShortCutMgr: TCnWizShortCutMgr;
begin

end;

{ TCnKeyBinding }

procedure TCnKeyBinding.BindKeyboard(
  const BindingServices: IOTAKeyBindingServices);
begin

end;

constructor TCnKeyBinding.Create(AOwner: TCnWizShortCutMgr);
begin

end;

destructor TCnKeyBinding.Destroy;
begin
  inherited;

end;

function TCnKeyBinding.GetBindingType: TBindingType;
begin

end;

function TCnKeyBinding.GetDisplayName: string;
begin

end;

function TCnKeyBinding.GetName: string;
begin

end;

procedure TCnKeyBinding.KeyProc(const Context: IOTAKeyContext;
  KeyCode: TShortcut; var BindingResult: TKeyBindingResult);
begin

end;

end.
