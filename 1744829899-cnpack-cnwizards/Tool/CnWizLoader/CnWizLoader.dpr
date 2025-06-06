{******************************************************************************}
{                       CnPack For Delphi/C++Builder                           }
{                     中国人自己的开放源码第三方开发包                         }
{                   (C)Copyright 2001-2025 CnPack 开发组                       }
{                   ------------------------------------                       }
{                                                                              }
{            本开发包是开源的自由软件，您可以遵照 CnPack 的发布协议来修        }
{        改和重新发布这一程序。                                                }
{                                                                              }
{            发布这一开发包的目的是希望它有用，但没有任何担保。甚至没有        }
{        适合特定目的而隐含的担保。更详细的情况请参阅 CnPack 发布协议。        }
{                                                                              }
{            您应该已经和开发包一起收到一份 CnPack 发布协议的副本。如果        }
{        还没有，可访问我们的网站：                                            }
{                                                                              }
{            网站地址：https://www.cnpack.org                                  }
{            电子邮件：master@cnpack.org                                       }
{                                                                              }
{******************************************************************************}

library CnWizLoader;
{* |<PRE>
================================================================================
* 软件名称：CnPack IDE 专家包
* 单元名称：CnWizard 专家 DLL 加载器实现单元
* 单元作者：CnPack 开发组 (master@cnpack.org)
* 备    注：
* 开发平台：PWin7 + Delphi 5.0
* 兼容测试：所有版本的 Delphi
* 本 地 化：该单元无需本地化
* 修改记录：2025.02.03 V1.1
*               重构，把部分公用内容独立以做 64 位适配
*           2020.05.13 V1.0
*               创建单元，持续根据 Delphi 的新版及新 Update 包及新 Patch 包更新
================================================================================
|</PRE>}

uses
  SysUtils,
  Classes,
  Windows,
  Forms,
  ToolsAPI,
  CnWizLoadUtils in 'CnWizLoadUtils.pas';

{$R *.RES}

// 加载器 DLL 初始化入口函数，加载对应版本的专家包 DLL
function InitWizard(const BorlandIDEServices: IBorlandIDEServices;
  RegisterProc: TWizardRegisterProc;
  var Terminate: TWizardTerminateProc): Boolean; stdcall;
var
  Dll: string;
  Entry: TWizardEntryPoint;
begin
  if FindCmdLineSwitch(SCnNoCnWizardsSwitch, ['/', '-'], True) then
  begin
    Result := True;
    Exit;
  end;

  Result := False;
  Dll := GetWizardDll;

  if (Dll <> '') and FileExists(Dll) then
  begin
    OutputDebugString(PChar(Format('Get DLL: %s', [Dll])));

    DllInst := LoadLibraryA(PAnsiChar(Dll));
    if DllInst <> 0 then
    begin
      Entry := TWizardEntryPoint(GetProcAddress(DllInst, WizardEntryPoint));
      if Assigned(Entry) then
      begin
        // 调用真正的 DLL 初始化，并接收其卸载过程的指针
        Result := Entry(BorlandIDEServices, RegisterProc, LoaderTerminateProc);
        // IDE 的卸载过程则指给我们的
        Terminate := LoaderTerminate;
      end
      else
        OutputDebugString(PChar(Format('DLL Corrupted! No Entry %s', [WizardEntryPoint])));
    end
    else
      OutputDebugString(PChar(Format('DLL Loading Error! %d', [GetLastError])));
  end
  else
    OutputDebugString(PChar(Format('DLL %s NOT Found!', [Dll])));
end;

exports
  InitWizard name WizardEntryPoint;

begin
end.
