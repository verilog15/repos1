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

unit CnWizSplash;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�IDE ���洰�ڴ���Ԫ
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
* ����ƽ̨��WinXpPro + Delphi 7
* ���ݲ��ԣ�PWin9X/2000/XP + Delphi 5/6/7 + C++Builder 5/6
* �� �� �����õ�Ԫ�е��ַ��������ϱ��ػ�����ʽ
* �޸ļ�¼��2007.03.31 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

{$I CnWizards.inc}

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls, CnWizManager, {$IFNDEF CNWIZARDS_MINIMUM} CnWizUtils, {$ENDIF}
  CnWizConsts;

procedure CnWizInitSplash;

implementation

{$IFDEF DEBUG}
uses
  CnDebug;
{$ENDIF}

const
  CnSplashScreenFormName = 'SplashScreen';

procedure CnWizInitSplash;
var
  I: Integer;
  SplashForm: TCustomForm;
  pnlCnWiz: TPanel;
  imgCnWiz: TImage;
begin
  SplashForm := nil;
  for I := 0 to Screen.CustomFormCount - 1 do
  begin
    if Screen.CustomForms[I].Name = CnSplashScreenFormName then
    begin
      SplashForm := Screen.CustomForms[I];
      Break;
    end;
  end;

  if SplashForm <> nil then
  begin
    //pnlCnWiz
    pnlCnWiz := TPanel.Create(SplashForm);

    //imgCnWiz
    imgCnWiz := TImage.Create(SplashForm);

    //pnlCnWiz
    pnlCnWiz.Name := 'pnlCnWiz';
    pnlCnWiz.Visible := False;
    pnlCnWiz.Parent := SplashForm;
    pnlCnWiz.Caption := '';
    pnlCnWiz.Width := 32;
    pnlCnWiz.Height := 32;
    pnlCnWiz.Left := SplashForm.Width - pnlCnWiz.Width - 16;
    pnlCnWiz.Top := 16;
    pnlCnWiz.BevelOuter := bvNone;

    //imgCnWiz
    imgCnWiz.Name := 'imgCnWiz';
    imgCnWiz.Visible := False;
    imgCnWiz.Parent := pnlCnWiz;
    imgCnWiz.Align := alClient;

{$IFDEF CNWIZARDS_MINIMUM}
    // Draw Pure Color or Text to avoid Load Icon
{$ELSE}
    CnWizLoadBitmap(imgCnWiz.Picture.Bitmap, SCnAboutBmp);
{$ENDIF}
    if not imgCnWiz.Picture.Bitmap.Empty then
    begin
//      ûɶ�ô�    
//      imgCnWiz.ShowHint := True;
//      imgCnWiz.Hint := Format('%s %s.%s Build %s', ['CnPack IDE Wizards',
//        SCnWizardMajorVersion, SCnWizardMinorVersion, SCnWizardBuildDate]);
      pnlCnWiz.Visible := True;
      imgCnWiz.Visible := True;
      pnlCnWiz.BringToFront;
      SplashForm.Update;
    end;
  end;
end;

initialization
{$IFNDEF BDS}
  @InitSplashProc := @CnWizInitSplash;
{$ENDIF}

{$IFDEF DEBUG}
  CnDebugger.LogMsg('Initialization Done: CnWizSplash.');
{$ENDIF}

end.
