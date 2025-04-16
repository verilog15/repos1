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

unit CnPngUtils;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ���������
* ��Ԫ���ƣ�Png ��ʽ֧�ֵ�Ԫ
* ��Ԫ���ߣ��ܾ��� zjy@cnpack.org
* ��    ע������ pngimage �Ѿ��� Embarcadero �չ����µ����Э���ƺ������������Ŀ
*           ��Դ��Ϊ�˱����Ȩ���⣬�˴��� D2010 ��ʹ�ùٷ��� pngimage ����һ��
*           DLL �����Ͱ汾�� IDE ������ʹ�ã����� D10.4 �±��� 64 λ�汾��
* ����ƽ̨��Win7 + Delphi 2010/10.4
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�ʹ����е��ַ����Ѿ����ػ�����ʽ
* �޸ļ�¼��2025.02.03 V1.2
*               ���� 64 λ�汾
*           2017.03.15 V1.1
*               �û��Ƶķ�ʽȡ�� Assign �Ա��ⲿ�� PNG8 ͼƬȫ�ڵ�����
*           2011.07.05 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

interface

uses
  Windows, SysUtils, Graphics, pngimage;

function CnConvertPngToBmp(PngFile, BmpFile: PAnsiChar): LongBool; stdcall;

function CnConvertBmpToPng(BmpFile, PngFile: PAnsiChar): LongBool; stdcall;

exports
  CnConvertPngToBmp,
  CnConvertBmpToPng;

implementation

function CnConvertPngToBmp(PngFile, BmpFile: PAnsiChar): LongBool;
var
  Png: TPngImage;
  Bmp: TBitmap;
begin
  Result := False;
  if not FileExists(string(PngFile)) then
    Exit;

  Png := nil;
  Bmp := nil;
  try
    Png := TPngImage.Create;
    Bmp := TBitmap.Create;
    Png.LoadFromFile(string(PngFile));

    // PNG24 �Լ���͸���� PNG8 ��Ӧ ptmNone
    // PNG8 ͸����Ӧ ptmBit
    // PNG32 ��Ӧ ptmPartial
    if Png.TransparencyMode = ptmPartial then
    begin
      Bmp.Assign(Png);
    end
    else
    begin
      // ĳЩ png8 ͼ�������ȫ�ڣ����ɻ��Ƶķ�ʽ
      Bmp.Height := Png.Height;
      Bmp.Width := Png.Width;
      Png.Draw(Bmp.Canvas, Bmp.Canvas.ClipRect);
    end;

    if not Bmp.Empty then
    begin
      Bmp.SaveToFile(string(BmpFile));
      Result := True;
    end;
  finally
    Png.Free;
    Bmp.Free;
  end;
end;

function CnConvertBmpToPng(BmpFile, PngFile: PAnsiChar): LongBool;
var
  Png: TPngImage;
  Bmp: TBitmap;
  i, j: Integer;
  p, p1, p2: PByteArray;
begin
  Result := False;
  if not FileExists(string(BmpFile)) then
    Exit;

  Png := nil;
  Bmp := nil;
  try
    Bmp := TBitmap.Create;
    Bmp.LoadFromFile(string(BmpFile));
    if Bmp.PixelFormat = pf32bit then
    begin
      Png := TPngImage.CreateBlank(COLOR_RGBALPHA, 8, Bmp.Width, Bmp.Height);
      for i := 0 to Bmp.Height - 1 do
      begin
        p := Bmp.ScanLine[i];
        p1 := Png.Scanline[i];
        p2 := Png.AlphaScanline[i];
        for j := 0 to Bmp.Width - 1 do
        begin
          p1[j * 3] := p[j * 4];
          p1[j * 3 + 1] := p[j * 4 + 1];
          p1[j * 3 + 2] := p[j * 4 + 2];
          p2[j] := p[j * 4 + 3];
        end;
      end;
    end
    else
    begin
      Png := TPngImage.Create;
      Png.Assign(Bmp);
    end;
    if not Png.Empty then
    begin
      Png.SaveToFile(string(PngFile));
      Result := True;
    end;
  finally
    Png.Free;
    Bmp.Free;
  end;
end;

end.
