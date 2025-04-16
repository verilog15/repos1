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

unit CnWizMultiLangFrame;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�ר�Ұ�������Ƶ�Ԫ Frame ����
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
* ��    ע��
* ����ƽ̨��PWin7 + Delphi 5.01
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2024.03.16 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses 
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  CnLangMgr;

type
  TCnTranslateFrame = class(TFrame)
  {* ӵ�з��빦�ܵ� TFrame �࣬�ɶ������룬�������ڷ��õ�����}
  private

  protected
    procedure LanguageChanged(Sender: TObject);
    procedure Translate;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
  end;

implementation

{$R *.DFM}

{$IFDEF DEBUG}
uses
  CnDebug;
{$ENDIF}

{ TCnTranslateFrame }

constructor TCnTranslateFrame.Create(AOwner: TComponent);
begin
  inherited;
  DisableAlign;
  try
    Translate;
  finally
    EnableAlign;
  end;
  CnLanguageManager.AddChangeNotifier(LanguageChanged);
end;

destructor TCnTranslateFrame.Destroy;
begin
  CnLanguageManager.RemoveChangeNotifier(LanguageChanged);
  inherited;
end;

procedure TCnTranslateFrame.LanguageChanged(Sender: TObject);
begin
{$IFDEF DEBUG}
  CnDebugger.LogMsg('TCnTranslateFrame.LanguageChanged');
{$ENDIF}
  DisableAlign;
  try
    CnLanguageManager.TranslateFrame(Self);
  finally
    EnableAlign;
  end;
end;

procedure TCnTranslateFrame.Translate;
begin
  if (CnLanguageManager <> nil) and (CnLanguageManager.LanguageStorage <> nil)
    and (CnLanguageManager.LanguageStorage.LanguageCount > 0) then
  begin
    Screen.Cursor := crHourGlass;
    try
      CnLanguageManager.TranslateFrame(Self);
    finally
      Screen.Cursor := crDefault;
    end;
  end;
end;

end.
