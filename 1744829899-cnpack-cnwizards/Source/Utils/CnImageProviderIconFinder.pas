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

unit CnImageProviderIconFinder;
{* |<PRE>
================================================================================
* ������ƣ����������ԡ�����༭����
* ��Ԫ���ƣ�www.IconFinder.com ����֧�ֵ�Ԫ
* ��Ԫ���ߣ��ܾ��� zjy@cnpack.org
* ��    ע��
* ����ƽ̨��Win7 + Delphi 7
* ���ݲ��ԣ�
* �� �� �����õ�Ԫ�ʹ����е��ַ����Ѿ����ػ�����ʽ
* �޸ļ�¼��2011.07.04 V1.0
*               ������Ԫ
================================================================================
|</PRE>}

{$I CnWizards.inc}

interface

{$DEFINE USE_MSXML}

uses
  Windows, SysUtils, Classes, Graphics, CnImageProviderMgr, CnInetUtils,
{$IFDEF CN_USE_MSXML}
  ActiveX, ComObj, msxml,
{$ELSE}
  OmniXML, OmniXMLUtils,
{$ENDIF}
  CnCommon, CnWizXmlUtils;

type
  TCnImageProviderIconFinder = class(TCnBaseImageProvider)
  protected
    function DoSearchImage(Req: TCnImageReqInfo): Boolean; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    class procedure GetProviderInfo(var DispName, HomeUrl: string); override;
    procedure OpenInBrowser(Item: TCnImageRespItem); override;
    function SearchIconset(Item: TCnImageRespItem; var Req: TCnImageReqInfo): Boolean; override;
  end;
  
implementation

{ TCnImageProvider_IconFinder }

constructor TCnImageProviderIconFinder.Create;
begin
  inherited;
  FItemsPerPage := 20;
  FFeatures := [pfOpenInBrowser, pfSearchIconset];
end;

destructor TCnImageProviderIconFinder.Destroy;
begin

  inherited;
end;

class procedure TCnImageProviderIconFinder.GetProviderInfo(var DispName,
  HomeUrl: string);
begin
  DispName := 'IconFinder.com';
  HomeUrl := 'http://www.iconfinder.com';
end;

function TCnImageProviderIconFinder.DoSearchImage(Req: TCnImageReqInfo): Boolean;
var
  Url, Text: string;
  Lic: Integer;
  Xml: IXMLDocument;
  Root, Node, Icon: IXMLNode;
  I, J, Size: Integer;
  Item: TCnImageRespItem;
begin
  Result := False;
  if Req.CommercialLicenses then
    Lic := 1
  else
    Lic := 0;
  Url := Format('http://www.iconfinder.com/xml/search/?q=%s&c=%d&p=%d&l=%d&min=%d&max=%d&api_key=7cb3bc9947285bc4b3a2f2d8bd20a3dd',
    [Req.Keyword, FItemsPerPage, Req.Page, Lic, Req.MinSize, Req.MaxSize]);
  Text := string(CnInet_GetString(Url));
  Xml := CreateXMLDoc;
  if Xml.LoadXML(Text) then
  begin
    Root := FindNode(Xml, 'results');
    if Root <> nil then
    begin
      for I := 0 to Root.ChildNodes.Length - 1 do
      begin
        Node := Root.ChildNodes.Item[I];
        if SameText(Node.NodeName, 'opensearch:totalResults') then
        begin
          FTotalCount := XMLStrToIntDef(Node.Text, 0);
          FPageCount := (FTotalCount + FItemsPerPage - 1) div FItemsPerPage;
        end
        else if SameText(Node.NodeName, 'iconmatches') then
        begin
          Result := True;
          for J := 0 to Node.ChildNodes.Length - 1 do
          begin
            Icon := Node.ChildNodes.Item[J];
            if SameText(Icon.NodeName, 'icon') then
            begin
              Size := GetNodeTextInt(Icon, 'size', 0);
              if (Size >= Req.MinSize) and (Size <= Req.MaxSize) then
              begin
                Item := Items.Add;
                Item.Size := Size;
                Item.Id := GetNodeTextStr(Icon, 'id', '');
                Item.Url := GetNodeTextStr(Icon, 'image', '');
                Item.Ext := _CnExtractFileExt(Item.Url);
              end;
            end;
          end;
        end;
      end;
    end;
  end;
end;

procedure TCnImageProviderIconFinder.OpenInBrowser(Item: TCnImageRespItem);
begin
  OpenUrl(Format('http://www.iconfinder.com/icondetails/%s/%d/', [Item.Id, Item.Size]));
end;

function TCnImageProviderIconFinder.SearchIconset(Item: TCnImageRespItem;
  var Req: TCnImageReqInfo): Boolean;
var
  Url, Text: string;
  Xml: IXMLDocument;
  Root, Node: IXMLNode;
begin
  Result := False;
  Url := Format('http://www.iconfinder.com/xml/icondetails/?id=%s&size=%d&api_key=7cb3bc9947285bc4b3a2f2d8bd20a3dd',
    [Item.Id, Item.Size]);
  Text := string(CnInet_GetString(Url));
  Xml := CreateXMLDoc;
  if Xml.LoadXML(Text) then
  begin
    Root := FindNode(Xml, 'icon');
    if Root <> nil then
    begin
      Node := FindNode(Root, 'iconsetid');
      if Node <> nil then
      begin
        Req.Keyword := 'iconset:' + Node.Text;
        Req.Page := 0;
        Result := True;
      end;
    end;
  end;
end;

initialization
  ImageProviderMgr.RegisterProvider(TCnImageProviderIconFinder);

end.
