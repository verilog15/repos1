unit DasmX86;
(*
The i80x86 disassembler registration module of the DCU32INT utility
by Alexei Hmelnov.
----------------------------------------------------------------------------
E-Mail: alex@icc.ru
http://hmelnov.icc.ru/DCU/
----------------------------------------------------------------------------

See the file "readme.txt" for more details.

------------------------------------------------------------------------
                             IMPORTANT NOTE:
This software is provided 'as-is', without any expressed or implied warranty.
In no event will the author be held liable for any damages arising from the
use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:
1. The origin of this software must not be misrepresented, you must not
   claim that you wrote the original software.
2. Altered source versions must be plainly marked as such, and must not
   be misrepresented as being the original software.
3. This notice may not be removed or altered from any source
   distribution.
*)
interface

uses
  DasmDefs,DasmUtil,{$IFNDEF XMLx86}DasmOpT{$ELSE}x86Dasm{$ENDIF};

procedure Set80x86Disassembler{$IFDEF I64}(I64: Boolean){$ENDIF};

implementation

{$IFDEF OpSem}
uses x86DasmSem;
{$ENDIF}

procedure Set80x86Disassembler{$IFDEF I64}(I64: Boolean){$ENDIF};
begin
 {$IFDEF I64}
  modeI64 := I64;
 {$ENDIF}
  SetDisassembler(ReadCommand, ShowCommand,CheckCommandRefs
    {$IFDEF OpSem},{$IFDEF XMLx86}GetXMLX86CommandOperations{$ELSE}Nil{$ENDIF}{$ENDIF});
end ;

end.
