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

unit CnPascalAST;
{* |<PRE>
================================================================================
* ������ƣ�CnPack IDE ר�Ұ�
* ��Ԫ���ƣ�Pascal ��������﷨�����ɵ�Ԫ
* ��Ԫ���ߣ�CnPack ������ master@cnpack.org
* ��    ע��ͬʱ֧�� Unicode �ͷ� Unicode ������
*           ��֧�� Attribute����֧��������������֧�� class �ڵ� var/const/type ��
*           ��֧�ַ��͡���֧������ var
*           ��֧�� asm������������ע�ͻ�ԭ�Ƚϵ�
* ����ƽ̨��2024.09.07 V1.4
*               ����� Attribute ��֧��
*           2023.07.29 V1.3
*               ����Զ����ַ�����֧��
*           2023.04.01 V1.2
*               �������ֶ�������������ʹ��
*           2022.10.16 V1.1
*               ������ɽ���
*           2022.09.24 V1.0
*               ������Ԫ��������ʵ�ֹ��ܻ���
================================================================================
|</PRE>}

interface

{$I CnPack.inc}

uses
  SysUtils, Classes, TypInfo, mPasLex, CnPasWideLex, CnTree, CnContainers, CnStrings;

type
  ECnPascalAstException = class(Exception);

{$IFDEF SUPPORT_WIDECHAR_IDENTIFIER}  // 2005 ����
  TCnGeneralPasLex = TCnPasWideLex;
  TCnGeneralLexBookmark = TCnPasWideBookmark;
{$ELSE}                               // 5 6 7
  TCnGeneralPasLex = TmwPasLex;
  TCnGeneralLexBookmark = TmwPasLexBookmark;
{$ENDIF}

  TCnPasNodeType = (
    cntInvalid,

    cntSpace,
    cntLineComment,
    cntBlockComment,
    cntCompDirective,
    cntCRLFInComment,

    cntAsm,

    cntComma,
    cntSemiColon,
    cntColon,
    cntSingleOps,
    cntRelOps,
    cntAddOps,
    cntMulOps,
    cntRange,
    cntHat,
    cntDot,

    cntSquareOpen,
    cntSquareClose,
    cntRoundOpen,
    cntRoundClose,

    cntAssign,
    cntAddress,

    cntInt,
    cntFloat,
    cntString,
    cntIdent,
    cntGuid,
    cntInherited,

    cntConst,
    cntIndex,
    cntRead,
    cntWrite,
    cntImplements,
    cntDefault,
    cntStored,
    cntNodefault,
    cntReadonly,
    cntWriteonly,

    cntProgram,
    cntLibrary,
    cntUnit,
    cntInterfaceSection,
    cntImplementationSection,
    cntInitializationSection,
    cntFinalizationSection,
    cntAsmBlock,

    cntIf,
    cntCase,
    cntRepeat,
    cntWhile,
    cntFor,
    cntWith,
    cntTry,
    cntRaise,
    cntGoto,

    cntElse,
    cntTo,
    cntDo,
    cntExcept,
    cntFinally,
    cntOn,
    cntThen,
    cntUntil,
    cntAt,
    cntCaseSelector,
    cntCaseLabel,
    cntOut,
    cntObject,

    cntUsesClause,
    cntUsesDecl,
    cntTypeSection,
    cntTypeDecl,

    // ����������֧��
    cntTypeParams,
    cntTypeParamDeclList,
    cntTypeParamDecl,
    cntTypeParamList,
    cntTypeParamIdentList,

    cntTypeKeyword,
    cntTypeID,
    cntRestrictedType,
    cntCommonType,

    cntEnumeratedList,
    cntEmumeratedIdent,
    cntVariantSection,

    cntArrayType,
    cntOrdinalType,
    cntSubrangeType,
    cntSetType,
    cntFileType,
    cntOf,
    cntStringType,
    cntProcedureType,
    cntClassType,
    cntClassBody,
    cntClassHeritage,
    cntClassField,
    cntClassConstantDecl,
    cntObjectType,
    cntInterfaceType,
    cntInterfaceHeritage,

    cntRecord,
    cntFieldList,
    cntFieldDecl,
    cntRecVariant,
    cntIdentList,

    cntConstSection,
    cntConstDecl,
    cntExportsSection,
    cntExportDecl,

    cntSetConstructor,
    cntSetElement,

    cntVisibility,
    cntProcedureHeading,
    cntFunctionHeading,
    cntProperty,
    cntPropertyInterface,
    cntPropertySpecifiers,
    cntPropertyParameterList,
    cntVarSection,
    cntVarDecl,
    cntTypedConstant,
    cntFormalParameters,
    cntFormalParam,

    cntProcedureFunctionDecl,
    cntProcedure,
    cntFunction,
//    cntConstructor,
//    cntDestructor,
    cntDirective,

    cntLabel,
    cntLabelId,
    cntStatememt,
    cntSimpleStatement,
    cntCompoundStatement,

    cntExpressionList,
    cntConstExpression,
    cntConstExpressionInType,
    cntArrayConstant,
    cntRecordConstant,
    cntRecordFieldConstant,

    cntExpression,
    cntSimpleExpression,
    cntDesignator,
    cntQualId,
    cntTerm,
    cntFactor,

    cntSingleAttribute,
    cntAttributeItem,

    cntBegin,
    cntEnd
  );
  TCnPasNodeTypes = set of TCnPasNodeType;

  TCnPasAstLeaf = class(TCnLeaf)
  {* Text ���Դ��Ӧ���ַ���}
  private
    FNodeType: TCnPasNodeType;
    FTokenKind: TTokenKind;
    FReturn: Boolean;
    FNoSpaceBehind: Boolean;
    FNoSpaceBefore: Boolean;
    FLinearPos: Cardinal;
    function GetItems(AIndex: Integer): TCnPasAstLeaf;
    procedure SetItems(AIndex: Integer; const Value: TCnPasAstLeaf);
    function GetParent: TCnPasAstLeaf;
  public
    property Parent: TCnPasAstLeaf read GetParent;
    property Items[AIndex: Integer]: TCnPasAstLeaf read GetItems write SetItems; default;

    function GetPascalCode: string;
    function GetCppCode: string;

    function ConvertString: string;
    function ConvertNumber: string;

    function ConvertQualId: string;

    property NodeType: TCnPasNodeType read FNodeType write FNodeType;
    {* �﷨���ڵ�����}
    property TokenKind: TTokenKind read FTokenKind write FTokenKind;
    {* Pascal Token ���ͣ�ע���еĽڵ㱾��û��ʵ�ʶ�Ӧ�� Token���� tkNone ����}
    property LinearPos: Cardinal read FLinearPos write FLinearPos;
    {* �ýڵ��Ӧ���ļ�Ҳ���ǽ������������λ�ã�ע��ֻ�ܴ��� Ansi ����}
    property Return: Boolean read FReturn write FReturn;
    {* �� Token ���Ƿ�Ӧ���У�Ĭ�ϲ���}
    property NoSpaceBehind: Boolean read FNoSpaceBehind write FNoSpaceBehind;
    {* �� Token ���Ƿ��޿ո�Ĭ����}
    property NoSpaceBefore: Boolean read FNoSpaceBefore write FNoSpaceBefore;
    {* �� Token ǰ�Ƿ��޿ո�Ĭ����}
  end;

  TCnPasAstTree = class(TCnTree)
  private
    function GetItems(AbsoluteIndex: Integer): TCnPasAstLeaf;
    function GetRoot: TCnPasAstLeaf;
  public
    function ReConstructPascalCode: string;

    function ConvertToHppCode: string;
    function ConvertToCppCode: string;

    property Root: TCnPasAstLeaf read GetRoot;
    property Items[AbsoluteIndex: Integer]: TCnPasAstLeaf read GetItems;
  end;

  TCnPasAstGenerator = class
  private
    FLex: TCnGeneralPasLex;
    FTree: TCnPasAstTree;
    FStack: TCnObjectStack;
    FCurrentRef: TCnPasAstLeaf;
    FReturnRef: TCnPasAstLeaf;
    FLocked: Integer;
    procedure Lock;
    procedure Unlock;
    function MatchCreateLeaf(AToken: TTokenKind; NodeType: TCnPasNodeType = cntInvalid): TCnPasAstLeaf;
    procedure MatchLeafStep(AToken: TTokenKind);
  protected
    procedure MarkReturnFlag(ALeaf: TCnPasAstLeaf);
    procedure MarkNoSpaceBehindFlag(ALeaf: TCnPasAstLeaf);
    procedure MarkNoSpaceBeforeFlag(ALeaf: TCnPasAstLeaf);

    procedure PushLeaf(ALeaf: TCnPasAstLeaf);
    procedure PopLeaf;

    function MatchCreateLeafAndPush(AToken: TTokenKind; NodeType: TCnPasNodeType = cntInvalid): TCnPasAstLeaf;
    // ����ǰ Token ����һ���ڵ㣬��Ϊ FCurrentRef �����һ���ӽڵ㣬�ٰ� FCurrentRef �����ջ������ȡ�� FCurrentRef
    function MatchCreateLeafAndStep(AToken: TTokenKind; NodeType: TCnPasNodeType = cntInvalid): TCnPasAstLeaf;
    // ����ǰ Token ����һ���ڵ㣬��Ϊ FCurrentRef �����һ���ӽڵ㣬��������������һ����Ч�ڵ�
    procedure NextToken;
    // Lex ��ǰ�н�����һ����Ч Token�������ע�ͣ������������Ȳ�������������ָ�
    procedure SkipComments;
    // ��ʼʱ���ã�����ע�͵�����Ч Token

    function ForwardToken(Step: Integer = 1): TTokenKind;
    // ȡ�� Step ����Ч Token ������ǰ�н����ڲ�ʹ����ǩ���лָ�
  public
    constructor Create(const Source: string); virtual;
    destructor Destroy; override;

    property Tree: TCnPasAstTree read FTree;
    {* Build ��Ϻ���﷨��}

    // ��Щ�﷨�����ǹؼ��ֿ�ͷ��֮����һ���ӽڵ����

    // ����Щ�����Ԫ���㣬�������������Ҫ�Ǹ��ڵ㣬Ԫ�����ӽڵ㣬���Ƿ�����Ҫ��������
    procedure Build;
    procedure BuildProgram;
    procedure BuildLibrary;
    procedure BuildUnit;

    procedure BuildProgramBlock;

    procedure BuildBlock;

    procedure BuildInterfaceSection;

    procedure BuildInterfaceDecl;

    procedure BuildImplementationSection;

    procedure BuildInitSection;

    procedure BuildDeclSection;
    procedure BuildLabelDeclSection;

    procedure BuildExportedHeading;
    {* ��װ�����뺯������������}

    procedure BuildExportsSection;
    procedure BuildExportsList;
    procedure BuildExportsDecl;

    procedure BuildProcedureDeclSection;
    {* ��װ�����뺯��ʵ���������ܰ��� class �ͺ�������������ͬ}
    procedure BuildProcedureFunctionDecl;
    {* ��װ���������ʵ���壬�ͺ�������������ͬ}

    // Build ϵ�к���ִ�����FLex ��Ӧ Next ��β��֮�����һ�� Token
    procedure BuildTypeSection;
    {* ���� type �ؼ���ʱ�����ã��½� type �ڵ㣬�����Ƕ�� typedecl �ӷֺţ�ÿ�� typedecl ���½ڵ�}
    procedure BuildTypeDecl;
    {* �� BuildTypeSection ѭ�����ã�ÿ������һ���ڵ㲢���������� typedecl �ڲ���Ԫ�ص��ӽڵ㣬�������ֺ�}

    procedure BuildTypeParams;
    {* �� BuildTypeDecl �ȵ��ã���Ϊ���͵ĳ���֧�֣��ڲ��� <BuildTypeParamDeclList>}
    procedure BuildTypeParamDeclList;
    {* �� BuildTypeParams ���ã���Ϊ���͵ĳ���֧�֣��ڲ��Ƿֺŷָ��� BuildTypeParamDecl}
    procedure BuildTypeParamDecl;
    {* �� BuildTypeParamDeclList ���ã���Ϊ���͵ĳ���֧�֣��ڲ��� BuildTypeParamList: IdentList ������ʽ}
    procedure BuildTypeParamList;
    {* �� BuildTypeParamDecl ���ã���Ϊ���͵ĳ���֧�֣��ڲ��Ƕ��ŷָ��� Ident<TypeParams>������ֿ��ܵ��� BuildTypeParams}

    procedure BuildTypeParamIdentList;
    {* ������ IdentList ���ڲ�ѭ�����õ��� BuildTypeParamIdent��Ŀǰ���������ڲ����ã�δ������кʹ�С�ںŻ��� }
    procedure BuildTypeParamIdent;
    {* ������ Ident ��ÿ�� Ident ��������ӷ��͵� <> ֧��}

    procedure BulidRestrictedType;
    {* ��������}
    procedure BuildCommonType;
    {* ������ͨ���ͣ���Ӧ Type}

    procedure BuildSimpleType;
    {* ���򵥵����ͣ�Subrange/Enum/Ident��һ���̶����ܱ� CommonType ����}
    
    procedure BuildEnumeratedType;
    {* ��װһ��ö�����ͣ�(a, b) ����}
    procedure BuildEnumeratedList;
    {* ��װһ��ö�������е��б�(a, b) �����е� a, b}
    procedure BuildEmumeratedIdent;
    {* ��װһ��ö�������еĵ���}

    procedure BuildStructType;
    procedure BuildArrayType;
    procedure BuildSetType;
    procedure BuildFileType;
    procedure BuildRecordType;
    procedure BuildProcedureType;
    procedure BuildPointerType;
    procedure BuildStringType;
    procedure BuildOrdinalType;
    procedure BuildSubrangeType;
    procedure BuildOrdIdentType;
    procedure BuildTypeID;
    procedure BuildGuid;

    procedure BuildClassType;
    procedure BuildClassBody;
    procedure BuildClassHeritage;
    procedure BuildClassMemberList;
    procedure BuildClassMembers;
    procedure BuildObjectType;
    procedure BuildInterfaceType;
    procedure BuildInterfaceHeritage;

    procedure BuildFieldList;
    procedure BuildClassVisibility;
    procedure BuildClassMethod;
    procedure BuildMethod;
    procedure BuildClassProperty;
    procedure BuildProperty;
    procedure BuildClassField;
    procedure BuildClassTypeSection;
    procedure BuildClassConstSection;
    procedure BuildClassConstantDecl;
    procedure BuildVarSection;
    procedure BuildVarDecl;
    procedure BuildTypedConstant;
    procedure BuildRecVariant;
    procedure BuildFieldDecl;
    procedure BuildVariantSection;

    procedure BuildPropertyInterface;
    procedure BuildPropertyParameterList;
    procedure BuildPropertySpecifiers;

    procedure BuildFunctionHeading;
    procedure BuildProcedureHeading;
    procedure BuildConstructorHeading;
    procedure BuildDestructorHeading;

    procedure BuildFormalParameters;
    {* ��װ�������̵Ĳ����б��������˵�С����}
    procedure BuildFormalParam;
    {* ��װ�������̵ĵ�������}

    procedure BuildConstSection;
    {* ��װ����������}
    procedure BuildConstDecl;
    {* ��װһ�������������������ֺ�}

    procedure BuildDirectives(NeedSemicolon: Boolean = True);
    {* ѭ����װ�� Diretives��NeedSemicolon ��ʾ�ڲ��Ƿ���ֺ�}

    procedure BuildDirective;
    {* ��װһ�� Directive��������ܸ�һ�����ʽ}

    procedure BuildUsesClause;
    {* ���� uses �ؼ���ʱ�����ã��½� uses �ڵ㣬�����Ƕ�� usesdecl �Ӷ��ţ�ÿ�� uses ���½ڵ�}
    procedure BuildUsesDecl;
    {* �� BuildUsesClause ѭ�����ã�ÿ������һ���ڵ㲢���������� usesdecl �ڲ���Ԫ�ص��ӽڵ�}
    
    procedure BuildSetConstructor;
    {* ��װһ�����ϱ��ʽ���г���ڵ�}
    procedure BuildSetElement;
    {* ��װһ������Ԫ��}

    procedure BulidAsmBlock;
    {* ��װ������}

    procedure BuildCompoundStatement;
    {* ��װһ��������䣬Ҳ���� begin��end �����������}
    procedure BuildStatementList;
    {* ��װһ�������ɵ�����б��Էֺŷָ�����������β���ķֺ�}
    procedure BuildStatement;
    {* ��װһ����䣬����Ϊ�ա�����ɿ��ܵ� Label ������ Simple �� Struct ������}

    procedure BuildLabelId;
    {* ��װһ�� LabelId}

    procedure BuildStructStatement;
    {* ��װ�ṹ��䣬�������£�}
    procedure BuildIfStatement;
    procedure BuildCaseStatement;
    procedure BuildRepeatStatement;
    procedure BuildWhileStatement;
    procedure BuildForStatement;
    procedure BuildWithStatement;
    procedure BuildTryStatement;
    procedure BuildRaiseStatement;

    procedure BuildCaseSelector;
    {* ��װ case ����е�ѡ����}
    procedure BuildCaseLabel;
    {* ��װ case ����е�ѡ�������� Label}
    procedure BuildExceptionHandler;
    {* ��װ try except ���е� on ���}

    procedure BuildSimpleStatement;
    {* ��װһ������䣬���� Designator��Designator ������ĸ�ֵ��inherited��Goto ��
      ע�⣬��俪ͷ�������С���ţ��޷�ֱ���ж��� Designator ������ (a)[0] := 1 ���֡�
            ���� SimpleStatement/Factor ������ (Caption := '') ����}
    procedure BuildExpressionList;
    {* ��װһ�����ʽ�б��ɶ��ŷָ�}
    procedure BuildExpression;
    {* ��װһ�����ʽ���ñ��ʽ�� SimpleExpression ��Ƚϡ��﷨�����ʵĶ�Ԫ���������}
    procedure BuildConstExpression;
    {* ��װһ�������ʽ�������ڱ��ʽ}
    procedure BuildConstExpressionInType;
    {* ��װһ���������еĳ������ʽ�������ڱ��ʽ�������ܳ��ֵȺŵ�}
    procedure BuildArrayConstant;
    procedure BuildRecordConstant;
    procedure BuildRecordFieldConstant;

    procedure BuildSimpleExpression;
    {* ��װһ���򵥱��ʽ����Ҫ�� Term ��ɣ�Term ֮���� AddOp ����}
    procedure BuildTerm;
    {* ��װһ�� Term����Ҫ�� Factor ��ɣ�Factor ֮���� MulOp ����}
    procedure BuildFactor;
    {* ��װһ�� Factor����Ȼ���������﷨���г��˼򵥱�ʶ�������򵥲��֣��г���ڵ�
      �����ڲ�ȴ�� Designator ���ֵ��߼���ʶ������λ�ͼ򵥱�ʶ����ͬ��@ ������}
    procedure BuildDesignator;
    {* ��װһ�� Designator ��ʶ������Ҫ�������ŵĶ�ά�����±ꡢ�Լ�С���ŵ� FunctionCall���Լ�ָ���ָ^ �Լ��� . �ͺ�������ʶ���������� @
      �����ָ�ܹ�ͨ���⼸�������������ȥ�ĵ��߼���ʶ�������Գ����� := ���󷽣���ͳ����� := �ҷ��� Expression �ǲ�ͬ��}
    procedure BuildQualId;
    {* ��װһ�� QualId����Ҫ�� Ident �Լ� (Designator as Type)����Ϊ Designator ����ʼ����}

    procedure BuildIdentList;
    {* ��װһ����ʶ���б����ŷָ�}
    procedure BuildIdent;
    {* ��װһ����ʶ�������Դ����}

    procedure BuildSingleAttribute;
    {* ��װһ���������ڵļ� Attribute}
    procedure BuildAttributeItem;
    {* ��װһ�� Attribute �һ������ Ident:Ident �� Ident(ExprList)��������ö��ŷֿ�}
  end;

function PascalAstNodeTypeToString(AType: TCnPasNodeType): string;

implementation

resourcestring
  SCnInvalidFileType = 'Invalid File Type!';
  SCnNotImplemented = 'NOT Implemented';
  SCnErrorStack = 'Stack Empty';
  SCnErrorNoMatchNodeType = 'No Matched Node Type';
  SCnErrorTokenNotMatchFmt = 'Token NOT Matched. Should %s, but meet %s: %s  Line %d Column %d';

const
  SpaceTokens = [tkCRLF, tkCRLFCo, tkSpace];

  CommentTokens = [tkSlashesComment, tkAnsiComment, tkBorComment];

  RelOpTokens = [tkGreater, tkLower, tkGreaterEqual, tkLowerEqual, tkNotEqual,
    tkEqual, tkIn, tkAs, tkIs];

  AddOPTokens = [tkPlus, tkMinus, tkOr, tkXor];

  MulOpTokens = [tkStar, tkDiv, tkSlash, tkMod, tkAnd, tkShl, tkShr];

  VisibilityTokens = [tkPublic, tkPublished, tkProtected, tkPrivate];

  ProcedureTokens = [tkProcedure, tkFunction, tkConstructor, tkDestructor];

  PropertySpecifiersTokens = [tkDispid, tkRead, tkIndex, tkWrite, tkStored,
    tkImplements, tkDefault, tkNodefault, tkReadonly, tkWriteonly];

  ClassMethodTokens = [tkClass] + ProcedureTokens;

  ClassMemberTokens = [tkIdentifier, tkClass, tkProperty, tkType, tkConst]
     + ProcedureTokens;  // ��֧�� class var/threadvar

  DirectiveTokens = [tkVirtual, tkOverride, tkAbstract, tkReintroduce, tkStdcall,
    tkCdecl, tkInline, tkName, tkIndex, tkLibrary, tkDefault, tkNoDefault,
    tkRead, tkReadonly, tkWrite, tkWriteonly, tkStored, tkImplements, tkOverload,
    tkPascal, tkRegister, tkExternal, tkAssembler, tkDynamic, tkAutomated,
    tkDispid, tkExport, tkFar, tkForward, tkNear, tkMessage, tkResident, tkSafecall,
    tkPlatform, tkDeprecated];
    // ���� platform, deprecated, unsafe, varargs ��һ��

  DirectiveTokensWithExpressions = [tkDispID, tkExternal, tkMessage, tkName,
    tkImplements, tkStored, tkRead, tkWrite, tkIndex];

  DeclSectionTokens = [tkClass, tkLabel, tkConst, tkResourcestring, tkType, tkVar,
    tkThreadvar, tkExports] + ProcedureTokens;

  InterfaceDeclTokens = [tkConst, tkResourcestring, tkThreadvar, tkType, tkVar,
    tkProcedure, tkFunction, tkExports];

  SimpleStatementTokens = [tkIdentifier, tkGoto, tkInherited,
    tkAddressOp, tkRoundOpen, tkVar, tkConst];
                              // 10.3 ���﷨���� inline var/const

  StructStatementTokens = [tkAsm, tkBegin, tkIf, tkCase, tkFor, tkWhile, tkRepeat,
    tkWith, tkTry, tkRaise];

  StatementTokens = [tkLabel] + SimpleStatementTokens + StructStatementTokens;

  CanBeIdentifierTokens = DirectiveTokens + [tkIdentifier]; // ���ֹؼ��ֿ�������������������

function PascalAstNodeTypeToString(AType: TCnPasNodeType): string;
begin
  Result := GetEnumName(TypeInfo(TCnPasNodeType), Ord(AType));

  if Length(Result) > 3 then
  begin
    Delete(Result, 1, 3);
    Result := UpperCase(Result);
  end;
end;

function NodeTypeFromToken(AToken: TTokenKind): TCnPasNodeType;
begin
  case AToken of
    // Goal
    tkProgram: Result := cntProgram;
    tkLibrary: Result := cntLibrary;
    tkUnit: Result := cntUnit;

    // Section
    tkUses: Result := cntUsesClause;
    tkType: Result := cntTypeSection;
    tkExports: Result := cntExportsSection;
    tkVar, tkThreadvar: Result := cntVarSection;
    tkImplementation: Result := cntImplementationSection;
    tkInitialization: Result := cntInitializationSection;
    tkFinalization: Result := cntFinalizationSection;

    // ����
    tkAsm: Result := cntAsm;
    tkBegin: Result := cntBegin;
    tkEnd: Result := cntEnd;
    tkProcedure, tkConstructor, tkDestructor: Result := cntProcedure;
    tkFunction: Result := cntFunction;

    // �ṹ�����
    tkIf: Result := cntIf;
    tkCase: Result := cntCase;
    tkRepeat: Result := cntRepeat;
    tkWhile: Result := cntWhile;
    tkFor: Result := cntFor;
    tkWith: Result := cntWith;
    tkTry: Result := cntTry;
    tkRaise: Result := cntRaise;
    tkGoto: Result := cntGoto;

    // �ṹ������ڲ�
    tkLabel: Result := cntLabel;
    tkElse: Result := cntElse;
    tkTo, tkDownto: Result := cntTo;
    tkDo: Result := cntDo;
    tkExcept: Result := cntExcept;
    tkFinally: Result := cntFinally;
    tkOn: Result := cntOn;
    tkThen: Result := cntThen;
    tkUntil: Result := cntUntil;
    tkAt: Result := cntAt;

    tkOut: Result := cntOut;
    tkObject: Result := cntObject;

    // Ԫ�أ�ע�͡�����ָ��
    tkBorComment, tkAnsiComment: Result := cntBlockComment;
    tkSlashesComment: Result := cntLineComment;
    tkCompDirect: Result := cntCompDirective;
    tkCRLFCo: Result := cntCRLFInComment;

    // Ԫ�أ���ʶ�������֡��ַ�����
    tkIdentifier, tkNil: Result := cntIdent;
    tkInteger, tkNumber: Result := cntInt; // ʮ��������������ͨ����
    tkFloat: Result := cntFloat;
    tkAsciiChar, tkString, tkMultiLineString: Result := cntString;
    tkInherited: Result := cntInherited;

    // Ԫ�أ��������������
    tkComma: Result := cntComma;
    tkSemiColon: Result := cntSemiColon;
    tkColon: Result := cntColon;
    tkDotDot: Result := cntRange;
    tkPoint: Result := cntDot;
    tkPointerSymbol: Result := cntHat;
    tkAssign: Result := cntAssign;
    tkAddressOp: Result := cntAddress;

    tkPlus, tkMinus, tkOr, tkXor: Result := cntAddOps;
    tkStar, tkDiv, tkSlash, tkMod, tkAnd, tkShl, tkShr: Result := cntMulOps;
    tkGreater, tkLower, tkGreaterEqual, tkLowerEqual, tkNotEqual, tkEqual, tkIn, tkAs, tkIs:
      Result := cntRelOps;
    tkNot: Result := cntSingleOps;

    tkSquareOpen: Result := cntSquareOpen;
    tkSquareClose: Result := cntSquareClose;
    tkRoundOpen: Result := cntRoundOpen;
    tkRoundClose: Result := cntRoundClose;

    // ����
    tkArray: Result := cntArrayType;
    tkSet: Result := cntSetType;
    tkFile: Result := cntFileType;
    tkKeyString: Result := cntStringType;
    tkOf: Result := cntOf;
    tkRecord, tkPacked: Result := cntRecord;
    tkInterface, tkDispinterface: Result := cntInterfaceType; // interface section �����ָ��
    tkClass: Result := cntClassType;

    // ����
    tkProperty: Result := cntProperty;
    tkConst, tkResourcestring: Result := cntConstSection;
    tkIndex: Result := cntIndex;  // TODO: ���Ե� Index Ҫ�� Directives �� index ����
    tkRead: Result := cntRead;
    tkWrite: Result := cntWrite;
    tkImplements: Result := cntImplements;
    tkDefault: Result := cntDefault;
    tkStored: Result := cntStored;
    tkNodefault: Result := cntNodefault;
    tkReadonly: Result := cntReadonly;
    tkWriteonly: Result := cntWriteonly;

    tkPrivate, tkProtected, tkPublic, tkPublished: Result := cntVisibility;
    tkVirtual, tkOverride, tkAbstract, tkReintroduce, tkStdcall, tkCdecl, tkInline, tkName,
    tkOverload, tkPascal, tkRegister, tkExternal, tkAssembler, tkDynamic, tkAutomated,
    tkDispid, tkExport, tkFar, tkForward, tkNear, tkMessage, tkResident, tkSafecall,
    tkPlatform, tkDeprecated:
      Result := cntDirective;
  else
    raise ECnPascalAstException.Create(SCnErrorNoMatchNodeType + ' '
      + GetEnumName(TypeInfo(TTokenKind), Ord(AToken)));
  end;
end;

{ TCnPasASTGenerator }

procedure TCnPasAstGenerator.Build;
begin
  SkipComments;

  case FLex.TokenID of
    tkProgram:
      BuildProgram;
    tkLibrary:
      BuildLibrary;
    tkUnit:
      BuildUnit;
  else
    raise ECnPascalAstException.Create(SCnInvalidFileType);
  end;
end;

procedure TCnPasAstGenerator.BuildArrayType;
begin
  MatchCreateLeafAndPush(tkArray);

  try
    if FLex.TokenID = tkSquareOpen then
    begin
      MatchCreateLeafAndPush(tkSquareOpen);
      try
        repeat
          BuildOrdinalType;
          if FLex.TokenID = tkComma then
            MatchCreateLeafAndStep(tkComma)
          else
            Break;
        until False;
      finally
        PopLeaf;
      end;
      MatchCreateLeafAndStep(tkSquareClose);
    end;

    MatchCreateLeafAndStep(tkOf);
    BuildCommonType; // Array �������ֻ���� Common Type����֧�� class ��
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassType;
begin
  MatchCreateLeafAndPush(FLex.TokenID);

  try
    if FLex.TokenID = tkSemiColon then // ǰ����������
      Exit;

    if FLex.TokenID = tkOf then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildIdent;
      Exit;
    end;

    if FLex.TokenID in [tkAbstract, tkSealed] then
      MatchCreateLeafAndStep(FLex.TokenID);

    BuildClassBody; // �ֺ��� TypeDecl �д���
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildConstExpression;
begin
  // �� BuildExpression ��ֻͬ�ǽڵ����Ͳ�ͬ
  MatchCreateLeafAndPush(tkNone, cntConstExpression);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� ConstExpression �ڵ�֮��

  try
    BuildSimpleExpression;
    while FLex.TokenID in RelOpTokens + [tkPoint, tkPointerSymbol, tkSquareOpen] do
    begin
      if FLex.TokenID in RelOpTokens then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildSimpleExpression;
      end
      else if FLex.TokenID = tkPointerSymbol then // ע�⣬�� . ^ [] ����չ������ԭʼ�﷨��û��
        MatchCreateLeafAndStep(FLex.TokenID)
      else if FLex.TokenID = tkPoint then
      begin
        MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));
        BuildIdent;
      end
      else if FLex.TokenID = tkSquareOpen then
      begin
        MatchCreateLeafAndPush(FLex.TokenID);
        try
          BuildExpressionList;
        finally
          PopLeaf;
        end;
        MatchCreateLeafAndStep(tkSquareClose);
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildDesignator;
begin
  MatchCreateLeafAndPush(tkNone, cntDesignator);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� Expression �ڵ�֮��

  try
    BuildQualId;
    while FLex.TokenID in [tkSquareOpen, tkRoundOpen, tkPoint, tkPointerSymbol] do
    begin
      case FLex.TokenID of
        tkSquareOpen: // �����±�
          begin
            MatchCreateLeafAndPush(tkSquareOpen);
            // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ�������Žڵ�֮��

            try
              BuildExpressionList;
            finally
              PopLeaf;
            end;
            MatchCreateLeafAndStep(tkSquareClose); // �ӽڵ���������һ�㣬�ٷ������׵���������
          end;
        tkRoundOpen: // Function Call
          begin
            MatchCreateLeafAndPush(tkRoundOpen);
            // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ�������Žڵ�֮��

            try
              BuildExpressionList;
            finally
              PopLeaf;
            end;
            MatchCreateLeafAndStep(tkRoundClose); // �ӽڵ���������һ�㣬�ٷ������׵���������
          end;
        tkPointerSymbol:
          begin
            MatchCreateLeafAndStep(FLex.TokenID);
          end;
        tkPoint:
          begin
            MarkNoSpaceBehindFlag(MatchCreateLeafAndStep(FLex.TokenID));
            BuildIdent;
          end;
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildEmumeratedIdent;
begin
  MatchCreateLeafAndPush(tkNone, cntEmumeratedIdent);

  try
    BuildIdent;
    if FLex.TokenID = tkEqual then
    begin
      MatchCreateLeafAndStep(tkEqual);
      BuildConstExpression;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildEnumeratedList;
begin
  MatchCreateLeafAndPush(tkNone, cntEnumeratedList);

  try
    repeat
      BuildEmumeratedIdent;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(tkComma)
      else
        Break;
    until False;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildEnumeratedType;
begin
  MatchCreateLeafAndPush(tkRoundOpen);

  try
    BuildEnumeratedList;
  finally
    PopLeaf;
  end;
  MatchCreateLeafAndStep(tkRoundClose);
end;

procedure TCnPasAstGenerator.BuildExpression;
begin
  MatchCreateLeafAndPush(tkNone, cntExpression);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� Expression �ڵ�֮��

  try
    BuildSimpleExpression;
    while FLex.TokenID in RelOpTokens + [tkPoint, tkPointerSymbol, tkSquareOpen] do
    begin
      if FLex.TokenID in RelOpTokens then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildSimpleExpression;
      end
      else if FLex.TokenID = tkPointerSymbol then // ע�⣬�� . ^ [] ����չ������ԭʼ�﷨��û��
        MatchCreateLeafAndStep(FLex.TokenID)
      else if FLex.TokenID = tkPoint then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildExpression;
      end
      else if FLex.TokenID = tkSquareOpen then
      begin
        MatchCreateLeafAndPush(FLex.TokenID);
        try
          BuildExpressionList;
        finally
          PopLeaf;
        end;
        MatchCreateLeafAndStep(tkSquareClose);
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildExpressionList;
begin
  MatchCreateLeafAndPush(tkNone, cntExpressionList);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� ExpressionList �ڵ�֮��

  try
    repeat
      BuildExpression;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(tkComma)
      else
        Break;
    until False;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFactor;
var
  T: TCnPasAstLeaf;
begin
  MatchCreateLeafAndPush(tkNone, cntFactor);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� Factor �ڵ�֮��

  try
    case FLex.TokenID of
      tkAddressOp:
        begin
          MarkNoSpaceBehindFlag(MatchCreateLeafAndPush(FLex.TokenID));
          // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ @ �ڵ�֮��

          try
            BuildDesignator;
          finally
            PopLeaf;
          end;
        end;
      tkIdentifier, tkNil, tkKeyString, tkIndex: // TODO: ���в��ֹؼ��ֿ�����������
        begin
          BuildDesignator;
          if FLex.TokenID = tkRoundOpen then
          begin
            MatchCreateLeafAndStep(tkRoundOpen);
            BuildExpressionList;
            MatchCreateLeafAndStep(tkRoundClose)
          end;
        end;
      tkAsciiChar, tkString: // AsciiChar �� #12 ���֣����Ժ� string ��ϣ������Ҫƴ�ճ�һ��
        begin
          T := MatchCreateLeafAndStep(FLex.TokenID);
          while FLex.TokenID in [tkAsciiChar, tkString] do
          begin
            if T <> nil then
              T.Text := T.Text + FLex.Token;
            NextToken;
          end;
        end;
      tkNumber, tkInteger, tkFloat, tkMultiLineString:
        MatchCreateLeafAndStep(FLex.TokenID);
      tkNot:
        begin
          MatchCreateLeafAndStep(FLex.TokenID);
          BuildFactor;
        end;
      tkSquareOpen:
        begin
          BuildSetConstructor;
        end;
      tkInherited:
        begin
          MatchCreateLeafAndPush(FLex.TokenID);
          // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ inherited �ڵ�֮��

          try
            BuildExpression;
          finally
            PopLeaf;
          end;
        end;
      tkRoundOpen:
        begin
          MatchCreateLeafAndPush(FLex.TokenID);
          // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ��С���Žڵ�֮��

          try
            BuildExpression;
          finally
            PopLeaf;
          end;
          MatchCreateLeafAndStep(tkRoundClose); // �ӽڵ���������һ�㣬�ٷ������׵���С����

          while FLex.TokenID = tkPointerSymbol do
            MatchCreateLeafAndStep(tkPointerSymbol)
        end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFileType;
begin
  MatchCreateLeafAndPush(tkFile);

  try
    if FLex.TokenID = tkOf then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildTypeID;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildIdent;
var
  T: TCnPasAstLeaf;
begin
  while FLex.TokenID = tkSquareOpen do // ���� Attribute ��֧��
    BuildSingleAttribute;

  if FLex.TokenID = tkAmpersand then
    MatchCreateLeafAndStep(FLex.TokenID);

  if FLex.TokenID = tkNil then // nil ��������
  begin
    MatchCreateLeafAndStep(FLex.TokenID);
    Exit;
  end;

  if FLex.TokenID in CanBeIdentifierTokens + [tkKeyString] then  // �����������������Ĺؼ���
  begin
    T := MatchCreateLeafAndStep(FLex.TokenID);
    if FLex.TokenID <> tkPoint then              // ����û����˳�
      Exit;

    if T <> nil then                             // �е�ͼӵ㲢����
      T.Text := T.Text + FLex.Token;
    NextToken;

    while FLex.TokenID in CanBeIdentifierTokens do
    begin
      if T <> nil then                           // ���б�����
        T.Text := T.Text + FLex.Token;
      NextToken;

      if FLex.TokenID <> tkPoint then            // ����û����˳�
        Exit;

      if T <> nil then                           // �е�ͼӵ㲢����
        T.Text := T.Text + FLex.Token;
      NextToken;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildLabelId;
begin
  if FLex.TokenID = tkInteger then
    MatchCreateLeafAndStep(tkInteger)
  else
    MatchCreateLeafAndStep(tkIdentifier);
end;

procedure TCnPasAstGenerator.BuildOrdinalType;
var
  Bookmark: TCnGeneralLexBookmark;
  IsRange: Boolean;

  procedure SkipOrdinalPrefix;
  begin
    repeat
      FLex.NextNoJunk;
    until not (FLex.TokenID in [tkIdentifier, tkPoint, tkInteger, tkString, tkRoundOpen, tkRoundClose,
      tkPlus, tkMinus, tkStar, tkSlash, tkDiv, tkMod]);
  end;

begin
  MatchCreateLeafAndPush(tkNone, cntOrdinalType);

  try
    if FLex.TokenID = tkRoundOpen then  // (a, b) ����
      BuildEnumeratedType
    else
    begin
      Lock;
      FLex.SaveToBookmark(Bookmark);

      try
        SkipOrdinalPrefix;
        IsRange := FLex.TokenID = tkDotDot;
      finally
        FLex.LoadFromBookmark(Bookmark);
        Unlock;
      end;

      if IsRange then
        BuildSubrangeType
      else
        BuildOrdIdentType;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildProcedureType;
begin
  MatchCreateLeafAndPush(tkNone, cntProcedureType);

  try
    if FLex.TokenID = tkProcedure then
    begin
      BuildProcedureHeading;
    end
    else if FLex.TokenID = tkFunction then
    begin
      BuildFunctionHeading;
    end;
    if FLex.TokenID = tkOf then
    begin
      MatchCreateLeafAndStep(tkOf);
      MatchCreateLeafAndStep(tkObject);
    end;

    BuildDirectives;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildQualId;
begin
  MatchCreateLeafAndPush(tkNone, cntQualId);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� QualId �ڵ�֮��

  try
    case FLex.TokenID of
      tkKeyString:
        MatchCreateLeafAndStep(FLex.TokenID); // TODO: ����һЩ�ؼ��ֿ�����ǿ������ת������������
      tkNil, tkIdentifier, tkIndex, tkAmpersand:           // TODO: ����һЩ�ؼ��ֿ�����������
        BuildIdent;
      tkRoundOpen:
        begin
          MatchCreateLeafAndPush(FLex.TokenID);
          // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ��С���Žڵ�֮��

          try
            BuildDesignator;
            if FLex.TokenID = tkAs then
            begin
              MatchCreateLeafAndStep(tkAs);
              BuildIdent; // TypeId ���� Ident
            end;
          finally
            PopLeaf;
          end;
          MatchCreateLeafAndStep(tkRoundClose); // �ӽڵ���������һ�㣬�ٷ������׵���С����
        end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildRecordType;
begin
  MatchCreateLeafAndPush(tkRecord);

  try
    if FLex.TokenID <> tkEnd then
      BuildFieldList;
    MatchCreateLeafAndStep(tkEnd);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSetConstructor;
begin
  MatchCreateLeafAndPush(tkNone, cntSetConstructor);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� SetConstructor �ڵ�֮��

  try
    MatchCreateLeafAndPush(tkSquareOpen);
   // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ�������Žڵ�֮��

    try
      while True do
      begin
        BuildSetElement;
        if FLex.TokenID = tkComma then
          MatchCreateLeafAndStep(tkComma)
        else
          Break;
      end;
    finally
      PopLeaf;
    end;
    MatchCreateLeafAndStep(tkSquareClose); // �ӽڵ���������һ�㣬�ٷ������׵���������
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSetElement;
begin
  MatchCreateLeafAndPush(tkNone, cntSetElement);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� SetElement �ڵ�֮��

  try
    BuildExpression;
    if FLex.TokenID = tkDotDot then
    begin
      MatchCreateLeafAndStep(tkDotDot);
      BuildExpression;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSetType;
begin
  MatchCreateLeafAndPush(tkNone, cntSetType);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� SetType �ڵ�֮��

  try
    MatchCreateLeafAndStep(tkSet);
    MatchCreateLeafAndStep(tkOf);
    BuildOrdinalType;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSimpleExpression;
begin
  MatchCreateLeafAndPush(tkNone, cntSimpleExpression);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� SimpleExpression �ڵ�֮��

  try
    if FLex.TokenID in [tkPlus, tkMinus, tkPointerSymbol] then
      MatchCreateLeafAndStep(FLex.TokenID);

    BuildTerm;
    while FLex.TokenID in AddOpTokens do
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildTerm;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSimpleStatement;
var
  Bookmark: TCnGeneralLexBookmark;
  IsDesignator: Boolean;
begin
  MatchCreateLeafAndPush(tkNone, cntSimpleStatement);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� SimpleStatement �ڵ�֮��

  try
    if FLex.TokenID = tkGoto then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildLabelId;
    end
    else if FLex.TokenID = tkInherited then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      // �������û�ˣ�Ҳ��������һ�� SimpleStatement
      if not (FLex.TokenID in [tkSemicolon, tkEnd, tkElse]) then
        BuildSimpleStatement;
    end
    else if FLex.TokenID = tkRoundOpen then
    begin
      // ( Statement ) ���֣��������� Designator ���ֿ�������Ҫ����취����
      FLex.SaveToBookmark(Bookmark);
      Lock;
      try
        // ��ǰ�ж��Ƿ� Designator
        try
          BuildDesignator;
          // ���� Designator ������ϣ��жϺ�����ɶ

          IsDesignator := FLex.TokenID in [tkAssign, tkRoundOpen, tkSemicolon,
            tkElse, tkEnd];
          // TODO: Ŀǰֻ�뵽�⼸����Semicolon ���� Designator �Ѿ���Ϊ��䴦�����ˣ�
          // else/end ����������û�ֺŵ����ж�ʧ��
        except
          IsDesignator := False;
          // ������������� := �����Σ�BuildDesignator �����
          // ˵�������Ǵ�����Ƕ�׵� Simplestatement
        end;
      finally
        Unlock;
        FLex.LoadFromBookmark(Bookmark);
      end;

      if IsDesignator then // �� Designator�������������еĸ�ֵ
      begin
        BuildDesignator;
        if FLex.TokenID = tkAssign then
        begin
          MatchCreateLeafAndStep(FLex.TokenID);
          BuildExpression;
        end;
      end
      else // �� ( Statement )
      begin
        MatchCreateLeafAndPush(tkRoundOpen);
        // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ��С���Žڵ�֮��

        try
          BuildSimpleStatement; // TODO: ��Ϊ Statement
        finally
          PopLeaf;
        end;
        MatchCreateLeafAndStep(tkRoundClose);
      end;
    end
    else // �� ( ��ͷ��Ҳ�� Designator�������������еĸ�ֵ
    begin
      BuildDesignator;
      if FLex.TokenID = tkAssign then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildExpression;
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildStringType;
begin
  MatchCreateLeafAndPush(tkNone, cntStringType);

  try
    if FLex.TokenID = tkKeyString then
      MatchCreateLeafAndStep(FLex.TokenID)
    else
      BuildIdent;

    if FLex.TokenID = tkRoundOpen then
    begin
      MatchCreateLeafAndPush(FLex.TokenID);
      try
        BuildExpression;
      finally
        PopLeaf;
      end;
      MatchCreateLeafAndStep(tkRoundClose);
    end
    else if FLex.TokenID = tkSquareOpen then
    begin
      MatchCreateLeafAndPush(FLex.TokenID);
      try
        BuildConstExpression;
      finally
        PopLeaf;
      end;
      MatchCreateLeafAndStep(tkSquareClose);
    end
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildStructType;
begin
  if FLex.TokenID = tkPacked then
    MatchCreateLeafAndStep(tkPacked);

  case FLex.TokenID of
    tkArray:
      BuildArrayType;
    tkSet:
      BuildSetType;
    tkFile:
      BuildFileType;
    tkRecord:
      BuildRecordType;
  end;
end;

procedure TCnPasAstGenerator.BuildTerm;
begin
  MatchCreateLeafAndPush(tkNone, cntTerm);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ����� Term �ڵ�֮��

  try
    BuildFactor;
    while FLex.TokenID in MulOpTokens do
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildFactor;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildCommonType;
var
  Bookmark: TCnGeneralLexBookmark;
  IsRange: Boolean;
begin
  MatchCreateLeafAndPush(tkNone, cntCommonType);

  try
    case FLex.TokenID of
      tkRoundOpen:
        begin
          BuildEnumeratedType;
        end;
      tkPacked, tkArray, tkSet, tkFile, tkRecord:
        begin
          BuildStructType;
        end;
      tkProcedure, tkFunction:
        begin
          BuildProcedureType;
        end;
      tkPointerSymbol:
        begin
          BuildPointerType;
        end;
    else
      if (FLex.TokenID = tkKeyString) or SameText(FLex.Token, 'String')
        or SameText(FLex.Token, 'AnsiString') or SameText(FLex.Token, 'WideString')
        or SameText(FLex.Token, 'UnicodeString') then
        BuildStringType
      else
      begin
        // TypeID? Խ��һ�� ConstExpressionInType ���Ƿ��� ..
        Lock;
        FLex.SaveToBookmark(Bookmark);

        try
          BuildConstExpressionInType;
          IsRange := FLex.TokenID = tkDotDot;
        finally
          FLex.LoadFromBookmark(Bookmark);
          UnLock;
        end;

        if IsRange then
          BuildSubrangeType
        else
          BuildTypeID;
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeDecl);
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ TypeDecl �ڵ�֮��

  try
    BuildIdent;

    if FLex.TokenID = tkLower then
      BuildTypeParams;

    MatchCreateLeafAndStep(tkEqual);
    if FLex.TokenID = tkType then
      MatchCreateLeafAndStep(tkType, cntTypeKeyword);

    // Ҫ�ֿ� RestrictType ����ͨ Type��ǰ�߰��� class/object/interface�����ֳ��ϲ��������
    if FLex.TokenID in [tkClass, tkObject, tkInterface, tkDispInterface] then
      BulidRestrictedType
    else
      BuildCommonType;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeParams;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeParams);
  try
    MatchCreateLeafAndStep(tkLower);

    BuildTypeParamDeclList;

    MatchCreateLeafAndStep(tkGreater);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeParamDeclList;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeParamDeclList);
  try
    BuildTypeParamDecl;
    while Flex.TokenID = tkSemicolon do
    begin
      MatchCreateLeafAndStep(tkSemicolon);
      BuildTypeParamDecl;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeParamDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeParamDecl);
  try
    BuildTypeParamList;
    while Flex.TokenID = tkColon do
    begin
      MatchCreateLeafAndStep(tkColon);
      BuildIdentList;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeParamList;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeParamList);
  try
    BuildIdent;

    // �������׷���
    if FLex.TokenID = tkLower then
      BuildTypeParams;

    while Flex.TokenID = tkComma do
    begin
      MatchCreateLeafAndStep(tkComma);
      BuildIdent;

      // �������׷���
      if FLex.TokenID = tkLower then
        BuildTypeParams;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeParamIdentList;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeParamIdentList);

  try
    repeat
      BuildTypeParamIdent;

      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(tkComma)
      else
        Break;
    until False;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeParamIdent;
var
  T: TCnPasAstLeaf;
begin
  if FLex.TokenID = tkAmpersand then
    MatchCreateLeafAndStep(FLex.TokenID);

  if FLex.TokenID = tkNil then // nil ��������û�з��͵�˵���������� &nil.DoSome �����
  begin
    MatchCreateLeafAndStep(FLex.TokenID);
    Exit;
  end;

  if FLex.TokenID in CanBeIdentifierTokens then  // �����������������Ĺؼ���
  begin
    T := MatchCreateLeafAndStep(FLex.TokenID);

    if FLex.TokenID = tkPoint then               // ����û���ȥ������
    begin
      if T <> nil then                             // �е�ͼӵ㲢����
        T.Text := T.Text + FLex.Token;
      NextToken;

      while FLex.TokenID in CanBeIdentifierTokens do
      begin
        if T <> nil then                           // ���б�����
          T.Text := T.Text + FLex.Token;
        NextToken;

        if FLex.TokenID <> tkPoint then            // ����û����˳�
          Break;

        if T <> nil then                           // �е�ͼӵ㲢����
          T.Text := T.Text + FLex.Token;
        NextToken;
      end;
    end;
  end;

  // ���Ϻ� BuildIdent �����ϸ���ȳ��� Exit ���֣���������һ������
  if FLex.TokenID = tkLower then
    BuildTypeParams;
end;

procedure TCnPasAstGenerator.BuildTypeSection;
begin
  MarkReturnFlag(MatchCreateLeafAndPush(tkType));
  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ type �ڵ�֮��

  try
    while FLex.TokenID in [tkIdentifier, tkAmpersand, tkSquareOpen] do
    begin
      BuildTypeDecl;
      MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildUsesClause;
begin
  if FLex.TokenID in [tkUses, tkRequires, tkContains] then
    MarkReturnFlag(MatchCreateLeafAndPush(FLex.TokenID));

  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ Uses �ڵ�֮��

  try
    while True do
    begin
      BuildUsesDecl;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(tkComma)
      else
        Break;
    end;

    MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildUsesDecl;
begin
  BuildIdent;
  if FLex.TokenID = tkIn then
  begin
    MatchCreateLeafAndStep(tkIn);
    MatchCreateLeafAndStep(tkString);
  end;
end;

procedure TCnPasAstGenerator.BulidRestrictedType;
begin
  MatchCreateLeafAndPush(tkNone, cntRestrictedType);

  try
    case FLex.TokenID of
      tkClass:
        BuildClassType;
      tkObject:
        BuildObjectType;
      tkInterface, tkDispinterface:
        BuildInterfaceType;
    end;
  finally
    PopLeaf;
  end;
end;

constructor TCnPasASTGenerator.Create(const Source: string);
begin
  inherited Create;
  FLex := TCnGeneralPasLex.Create;
  FStack := TCnObjectStack.Create;
  FTree := TCnPasAstTree.Create(TCnPasAstLeaf);
  FCurrentRef := FTree.Root as TCnPasAstLeaf;

  FLex.Origin := PChar(Source);
end;

destructor TCnPasASTGenerator.Destroy;
begin
  FTree.Free;
  FStack.Free;
  FLex.Free;
  inherited;
end;

procedure TCnPasAstGenerator.Lock;
begin
  Inc(FLocked);
end;

function TCnPasAstGenerator.MatchCreateLeafAndStep(AToken: TTokenKind;
  NodeType: TCnPasNodeType): TCnPasAstLeaf;
begin
  Result := MatchCreateLeaf(AToken, NodeType);
  MatchLeafStep(AToken);
end;

function TCnPasAstGenerator.MatchCreateLeaf(AToken: TTokenKind;
  NodeType: TCnPasNodeType): TCnPasAstLeaf;
begin
  Result := nil;
  if (AToken <> tkNone) and (AToken <> FLex.TokenID) then
  begin
{$IFDEF SUPPORT_WIDECHAR_IDENTIFIER}
    raise ECnPascalAstException.CreateFmt(SCnErrorTokenNotMatchFmt,
      [GetEnumName(TypeInfo(TTokenKind), Ord(AToken)),
       GetEnumName(TypeInfo(TTokenKind), Ord(FLex.TokenID)),
       FLex.Token, FLex.LineNumber + 1, FLex.TokenPos - FLex.LineStartOffset]);
{$ELSE}
    raise ECnPascalAstException.CreateFmt(SCnErrorTokenNotMatchFmt,
      [GetEnumName(TypeInfo(TTokenKind), Ord(AToken)),
       GetEnumName(TypeInfo(TTokenKind), Ord(FLex.TokenID)),
       FLex.Token, FLex.LineNumber + 1, FLex.TokenPos - FLex.LinePos]);
{$ENDIF}
  end;

  if NodeType = cntInvalid then
    NodeType := NodeTypeFromToken(AToken);

  if FLocked = 0 then // δ���Ŵ����ڵ�
  begin
    // �ô������нڵ�Ĵ�����
    if (FCurrentRef <> nil) and (FTree.Root <> FCurrentRef) then
      Result := FTree.AddChild(FCurrentRef) as TCnPasAstLeaf
    else
      Result := FTree.AddChild(FTree.Root) as TCnPasAstLeaf;

    Result.TokenKind := AToken;
    Result.NodeType := NodeType;
    Result.LinearPos := FLex.RunPos;

    if AToken <> tkNone then      // δ���Ÿ�ֵ
      Result.Text := FLex.Token;
  end;
end;

procedure TCnPasAstGenerator.MatchLeafStep(AToken: TTokenKind);
begin
  if AToken <> tkNone then // �����ݵ�ʵ�ʽڵ㣬�Ų���һ�£�����������Ҫǰ��
    NextToken;
end;

function TCnPasAstGenerator.MatchCreateLeafAndPush(AToken: TTokenKind;
  NodeType: TCnPasNodeType): TCnPasAstLeaf;
begin
  Result := MatchCreateLeaf(AToken, NodeType);
  if Result <> nil then
  begin
    PushLeaf(FCurrentRef);
    FCurrentRef := Result;  // Pop ֮ǰ���ڲ���ӵĽڵ��Ϊ�ýڵ�֮��
  end;
  MatchLeafStep(AToken);    // �� FCurrent ���ٲ��������� Step ���ע�͹Ҵ�ڵ�
end;

procedure TCnPasAstGenerator.NextToken;
begin
  repeat
    FLex.Next;

    if FLex.TokenID in CommentTokens + [tkCompDirect, tkCRLFCo] then
      MatchCreateLeaf(FLex.TokenID); // ���������ɱ�ѭ������������ tkCRLFCo ��Ϊ���ýڵ��е�ע���ڲ�����ʧ�س�����

  until not (FLex.TokenID in SpaceTokens + CommentTokens + [tkCompDirect]);
end;

function TCnPasAstGenerator.ForwardToken(Step: Integer): TTokenKind;
var
  Cnt: Integer;
  Bookmark: TCnGeneralLexBookmark;
begin
  FLex.SaveToBookmark(Bookmark);

  Cnt := 0;
  try
    while True do
    begin
      NextToken;
      Inc(Cnt);
      Result := FLex.TokenID;

      if Cnt >= Step then
        Exit;
    end;
  finally
    FLex.LoadFromBookmark(Bookmark);
  end;
end;

procedure TCnPasAstGenerator.PopLeaf;
begin
  if FLocked > 0 then // ����ʱ�� Pop����Ϊ Push Ҳ����
    Exit;

  if FStack.Count <= 0 then
    raise ECnPascalAstException.Create(SCnErrorStack);

  FCurrentRef := TCnPasAstLeaf(FStack.Pop);
end;

procedure TCnPasAstGenerator.PushLeaf(ALeaf: TCnPasAstLeaf);
begin
  if ALeaf <> nil then
    FStack.Push(ALeaf);
end;

procedure TCnPasAstGenerator.Unlock;
begin
  Dec(FLocked);
end;

procedure TCnPasAstGenerator.BuildInterfaceType;
begin
  MatchCreateLeafAndPush(FLex.TokenID);

  try
    if FLex.TokenID = tkSemiColon then // ǰ�������������ֺ����ⲿ����
      Exit;

    if FLex.TokenID = tkRoundOpen then
      BuildInterfaceHeritage;

    if FLex.TokenID = tkSquareOpen then
      BuildGuid;

    while FLex.TokenID in VisibilityTokens + ProcedureTokens + [tkProperty, tkSquareOpen] do
    begin
      while FLex.TokenID = tkSquareOpen do // ���� Attribute ��֧��
        BuildSingleAttribute;

      if FLex.TokenID in VisibilityTokens then
        BuildClassVisibility
      else if FLex.TokenID in ProcedureTokens then
        BuildMethod  // ע�ⲻ�� ClassMethod����Ϊ�ӿڲ�֧�� class function ����
      else if Flex.TokenID = tkProperty then
        BuildProperty;
    end;
    MatchCreateLeafAndStep(tkEnd);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildObjectType;
begin
  raise ECnPascalAstException.Create(SCnNotImplemented);
end;

procedure TCnPasAstGenerator.BuildOrdIdentType;
begin
  BuildIdent;
end;

procedure TCnPasAstGenerator.BuildSubrangeType;
begin
  MatchCreateLeafAndPush(tkNone, cntSubrangeType);

  try
    BuildConstExpression;
    MatchCreateLeafAndStep(tkDotDot);
    BuildConstExpression;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildPointerType;
begin
  MatchCreateLeafAndPush(tkPointerSymbol, cntHat);

  try
    BuildTypeID;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypeID;
begin
  MatchCreateLeafAndPush(tkNone, cntTypeID);

  try
    if FLex.TokenID in [tkKeyString, tkFile, tkConst, tkProcedure, tkFunction] then // BuildIdent �ڲ����Ϲؼ��� string��File
      MatchCreateLeafAndStep(FLex.TokenID)
    else
      BuildIdent;

    if FLex.TokenID = tkRoundOpen then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildExpression;
      MatchCreateLeafAndStep(tkRoundClose)
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFieldList;
begin
  MatchCreateLeafAndPush(tkNone, cntFieldList);

  try
    while not (FLex.TokenID in [tkEnd, tkCase, tkRoundClose]) do
    begin
      if FLex.TokenID in VisibilityTokens then
        BuildClassVisibility;

      while FLex.TokenID = tkSquareOpen do // ���� Attribute ��֧��
        BuildSingleAttribute;

      if FLex.TokenID = tkCase then
        Break
      else if FLex.TokenID in ProcedureTokens then
        BuildMethod
      else if FLex.TokenID = tkProperty then
        BuildProperty
      else if FLex.TokenID = tkType then
        BuildClassTypeSection
      else if FLex.TokenID = tkConst then
        BuildClassConstSection
      else if FLex.TokenID in [tkVar, tkThreadVar] then
        BuildVarSection
      else if FLex.TokenID <> tkEnd then
      begin
        BuildFieldDecl;
        if FLex.TokenID = tkSemiColon then
          MatchCreateLeafAndStep(tkSemiColon);
      end;
    end;

    // ���� case �ɱ���
    if FLex.TokenID = tkCase then
      BuildVariantSection;

    if FLex.TokenID = tkSemiColon then
      MatchCreateLeafAndStep(tkSemiColon);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildIdentList;
begin
  MatchCreateLeafAndPush(tkNone, cntIdentList);

  try
    repeat
      BuildIdent;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(tkComma)
      else
        Break;
    until False;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFunctionHeading;
begin
  MatchCreateLeafAndPush(tkFunction);

  try
    if FLex.TokenID in [tkIdentifier, tkAmpersand] then
    begin
      BuildTypeParamIdent;
      if FLex.TokenID = tkPoint then
      begin
        MatchCreateLeafAndStep(tkPoint);
        BuildTypeParamIdent;
      end;
    end;

    if FLex.TokenID = tkRoundOpen then
      BuildFormalParameters;

    MatchCreateLeafAndStep(tkColon);
    BuildCommonType;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildProcedureHeading;
begin
  if FLex.TokenID in [tkProcedure, tkConstructor, tkDestructor] then
    MatchCreateLeafAndPush(FLex.TokenID);

  try
    if FLex.TokenID in [tkIdentifier, tkAmpersand] then
    begin
      BuildTypeParamIdent;
      if FLex.TokenID = tkPoint then
      begin
        MatchCreateLeafAndStep(tkPoint);
        BuildTypeParamIdent;
      end;
    end;

    if FLex.TokenID = tkRoundOpen then
      BuildFormalParameters;

    if FLex.TokenID = tkEqual then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildIdent;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassConstSection;
begin
  MatchCreateLeafAndPush(tkConst);

  try
    while FLex.TokenID in [tkIdentifier, tkAmpersand] do
      BuildClassConstantDecl;

    MatchCreateLeafAndStep(tkSemiColon);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildMethod;
begin
  case FLex.TokenID of
    tkProcedure:
      BuildProcedureHeading;
    tkFunction:
      BuildFunctionHeading;
    tkConstructor:
      BuildConstructorHeading;
    tkDestructor:
      BuildDestructorHeading;
  end;

  if FLex.TokenID = tkSemiColon then
    MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon)); // ������ķֺ�

  BuildDirectives; // ����Ҫ��ѷֺ�Ҳ�Ե�
end;

procedure TCnPasAstGenerator.BuildClassTypeSection;
begin
  MarkReturnFlag(MatchCreateLeafAndPush(tkType));

  try
    while FLex.TokenID in [tkIdentifier, tkAmpersand] do
    begin
      BuildTypeDecl; // ������ BuildTypeSection������֮
      MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassVisibility;
begin
  MatchCreateLeafAndStep(FLex.TokenID);
end;

procedure TCnPasAstGenerator.BuildVariantSection;
var
  Bookmark: TCnGeneralLexBookmark;
  HasColon: Boolean;
begin
  MatchCreateLeafAndPush(tkCase, cntVariantSection);

  try
    Lock;
    FLex.SaveToBookmark(Bookmark);

    try
      BuildIdent;
      HasColon := FLex.TokenID = tkColon;
    finally
      FLex.LoadFromBookmark(Bookmark);
      Unlock;
    end;

    if HasColon then
    begin
      BuildIdent;
      MatchCreateLeafAndStep(tkColon);
      BuildTypeID;
    end
    else
      BuildTypeID;

    MatchCreateLeafAndStep(tkOf);
    repeat
      BuildRecVariant;
      if FLex.TokenID = tkSemiColon then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        if FLex.TokenID in [tkEnd, tkRoundClose] then
          Break;
      end
      else
        Break;
    until False;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildVarSection;
begin
  if FLex.TokenID in [tkVar, tkThreadvar] then
    MarkReturnFlag(MatchCreateLeafAndPush(FLex.TokenID));

  try
    while FLex.TokenID in [tkIdentifier, tkAmpersand] do
    begin
      BuildVarDecl;
      MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFieldDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntFieldDecl);

  try
    BuildIdentList;
    MatchCreateLeafAndStep(tkColon);
    BuildCommonType;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildProperty;
begin
  MatchCreateLeafAndPush(tkProperty);

  try
    BuildIdent;
    if FLex.TokenID in [tkSquareOpen, tkColon] then
      BuildPropertyInterface;
    BuildPropertySpecifiers;
  finally
    PopLeaf;
  end;

  FReturnRef := MatchCreateLeafAndStep(tkSemiColon);

  if FLex.TokenID = tkDefault then
  begin
    MatchCreateLeafAndStep(FLex.TokenID);
    FReturnRef := MatchCreateLeafAndStep(tkSemiColon);
  end;
  FReturnRef.Return := True;
end;

procedure TCnPasAstGenerator.BuildRecVariant;
begin
  MatchCreateLeafAndPush(tkNone, cntRecVariant);

  try
    repeat
      BuildConstExpression;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(tkComma)
      else
        Break;
    until False;

    MatchCreateLeafAndStep(tkColon);
    if FLex.TokenID = tkRoundOpen then
    begin
      MatchCreateLeafAndPush(tkRoundOpen);

      try
        BuildFieldList;
      finally
        PopLeaf;
      end;
      MatchCreateLeafAndStep(tkRoundClose);
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildPropertyInterface;
begin
  MatchCreateLeafAndPush(tkNone, cntPropertyInterface);

  try
    if FLex.TokenID <> tkColon then
      BuildPropertyParameterList;
    MatchCreateLeafAndStep(tkColon);
    BuildCommonType;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildPropertySpecifiers;
var
  ID: TTokenKind;
begin
  MatchCreateLeafAndPush(tkNone, cntPropertySpecifiers);

  try
    while FLex.TokenID in PropertySpecifiersTokens do
    begin
      ID := FLex.TokenID;
      MatchCreateLeafAndStep(FLex.TokenID);
      case ID of
        tkDispid:
          begin
            BuildExpression;
          end;
        tkIndex, tkStored, tkDefault:
          begin
            BuildConstExpression;
          end;
        tkRead, tkWrite:
          begin
            BuildDesignator;
          end;
        tkImplements:
          begin
            BuildTypeID;
          end;
        // tkNodefault, tkReadonly, tkWriteonly ֱ�� Match ��
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildPropertyParameterList;
begin
  MatchCreateLeafAndPush(tkNone, cntPropertyParameterList);

  try
    MatchCreateLeafAndPush(tkSquareOpen);

    try
      repeat
        if FLex.TokenID in [tkVar, tkConst, tkOut] then
          MatchCreateLeafAndStep(FLex.TokenID); // TODO: ���� var/const ���ε������� VarSection��ConstSection

        BuildIdentList;
        MatchCreateLeafAndStep(tkColon);
        BuildTypeID;

        if FLex.TokenID <> tkSemiColon then
          Break;
      until False;
    finally
      PopLeaf;
    end;

    MatchCreateLeafAndStep(tkSquareClose);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildVarDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntVarDecl);

  try
    BuildIdentList;
    if FLex.TokenID = tkColon then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildCommonType;
    end;

    if FLex.TokenID = tkEqual then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildTypedConstant;
    end
    else if FLex.TokenID = tkAbsolute then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildConstExpression; // ���� Ident ������
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTypedConstant;
type
  TCnTypedConstantType = (tcConst, tcArray, tcRecord);
var
  TypedConstantType: TCnTypedConstantType;
  Bookmark: TCnGeneralLexBookmark;
begin
  MatchCreateLeafAndPush(tkNone, cntTypedConstant);

  try
    if FLex.TokenID = tkSquareOpen then
    begin
      BuildSetConstructor;
      while FLex.TokenID in (AddOPTokens + MulOPTokens) do
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildSetConstructor;
      end;
    end
    else if FLex.TokenID = tkRoundOpen then
    begin
      // TODO: �ж������鳣�����ǽṹ����
      TypedConstantType := tcConst;
      if ForwardToken = tkRoundOpen then 
      begin
        // ������滹�����ţ���˵���������ǳ����� array�������ж�
        Lock;
        FLex.SaveToBookmark(Bookmark);

        try
          try
            BuildConstExpression;

            if FLex.TokenID = tkComma then
              TypedConstantType := tcArray
            else if FLex.TokenID = tkSemiColon then
              TypedConstantType := tcConst;
          except
            // ��������������������
            TypedConstantType := tcArray;
          end;
        finally
          FLex.LoadFromBookmark(Bookmark);
          Unlock;
        end;
      end
      else // �����һ������
      begin
        // �ж����ź��Ƿ� a: 0 ������ʽ�����ź�����ð�ű�ʾ�ǽṹ
        if (ForwardToken() = tkIdentifier) and (ForwardToken(2) = tkColon) then
          TypedConstantType := tcRecord
        else
        begin
          // �������ж� ( ConstExpr[, ConstExpr] ); ���֣��ж��š���û���ŵ��������źͷֺţ���������
          Lock;
          FLex.SaveToBookmark(Bookmark);

          try
            MatchCreateLeafAndStep(tkRoundOpen);
            try
              BuildConstExpression;
              if FLex.TokenID = tkComma then // (1, 1) ������
                TypedConstantType := tcArray;
              if FLex.TokenID = tkRoundClose then
                MatchCreateLeafAndStep(FLex.TokenID);

              if FLex.TokenID = tkSemicolon then // (1) ������
                TypedConstantType := tcArray;
            except
              ;
            end;
          finally
            FLex.LoadFromBookmark(Bookmark);
            Unlock;
          end;
        end;
      end;

      if TypedConstantType = tcArray then
        BuildArrayConstant
      else if TypedConstantType = tcRecord then
        BuildRecordConstant
      else
        BuildConstExpression;
    end
    else
      BuildConstExpression;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildInterfaceHeritage;
begin
  MatchCreateLeafAndPush(tkNone, cntInterfaceHeritage);

  try
    MatchCreateLeafAndPush(tkRoundOpen);
    try
      BuildIdentList;
    finally
      PopLeaf;
    end;
    MarkReturnFlag(MatchCreateLeafAndStep(tkRoundClose));
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildGuid;
begin
  MatchCreateLeafAndPush(tkNone, cntGuid);

  try
    MatchCreateLeafAndPush(tkSquareOpen);
    try
      MatchCreateLeafAndStep(tkString); // ������һ���ַ���
    finally
      PopLeaf;
    end;
    MatchCreateLeafAndStep(tkSquareClose);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassMethod;
begin
  if FLex.TokenID = tkClass then
    MatchCreateLeafAndStep(FLex.TokenID);
  BuildMethod;
end;

procedure TCnPasAstGenerator.BuildClassProperty;
begin
  if FLex.TokenID = tkClass then
    MatchCreateLeafAndStep(FLex.TokenID);
  BuildProperty;
end;

procedure TCnPasAstGenerator.BuildConstructorHeading;
begin
  MatchCreateLeafAndPush(tkConstructor);

  try
    BuildIdent;
    if FLex.TokenID = tkRoundOpen then
      BuildFormalParameters;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildDestructorHeading;
begin
  MatchCreateLeafAndPush(tkDestructor);

  try
    BuildIdent;
    if FLex.TokenID = tkRoundOpen then
      BuildFormalParameters;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFormalParameters;
begin
  MatchCreateLeafAndPush(tkNone, cntFormalParameters);

  try
    MatchCreateLeafAndPush(tkRoundOpen);

    try
      if FLex.TokenID <> tkRoundClose then
      begin
        repeat
          BuildFormalParam;
          if FLex.TokenID = tkSemiColon then
            MatchCreateLeafAndStep(FLex.TokenID)
          else
            Break;
        until False;
      end;
    finally
      PopLeaf;
    end;
    MatchCreateLeafAndStep(tkRoundClose);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildFormalParam;
begin
  MatchCreateLeafAndPush(tkNone, cntFormalParam);

  try
    while FLex.TokenID = tkSquareOpen do // ���� Attribute ��֧��
      BuildSingleAttribute;

    if FLex.TokenID in [tkVar, tkConst, tkOut] then
      MatchCreateLeafAndStep(FLex.TokenID);
    BuildIdentList;

    if FLex.TokenID = tkColon then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      if FLex.TokenID = tkArray then
      begin
        MatchCreateLeafAndStep(tkArray); // �����������в�������� array[0..1] ����
        MatchCreateLeafAndStep(tkOf);

        if FLex.TokenID in [tkKeyString, tkFile, tkConst] then // array of const ����
          MatchCreateLeafAndStep(FLex.TokenID);

        if FLex.TokenID = tkRoundOpen then
        begin
          MatchCreateLeafAndPush(FLex.TokenID);
          try
            BuildSubrangeType;
          finally
            PopLeaf;
          end;
          MatchCreateLeafAndStep(tkRoundClose);
        end
        else
        begin
          BuildConstExpression;
          if FLex.TokenID = tkDotDot then
          begin
            MatchCreateLeafAndStep(Flex.TokenID);
            BuildConstExpression;
          end;
        end;
      end
      else if FLex.TokenID in [tkIdentifier, tkKeyString, tkFile] then
        BuildCommonType;

      if FLex.TokenID = tkEqual then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildConstExpression;
      end;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassHeritage;
begin
  MatchCreateLeafAndPush(tkNone, cntClassHeritage);

  try
    MarkNoSpaceBeforeFlag(MatchCreateLeafAndPush(tkRoundOpen));
    try
      BuildTypeParamIdentList;
    finally
      PopLeaf;
    end;
    MarkReturnFlag(MatchCreateLeafAndStep(tkRoundClose));
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassBody;
begin
  MatchCreateLeafAndPush(tkNone, cntClassBody);

  try
    if FLex.TokenID = tkRoundOpen then
      BuildClassHeritage;

    if FLex.TokenID <> tkSemiColon then
    begin
      BuildClassMemberList;
      MatchCreateLeafAndStep(tkEnd);
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildClassMemberList;
var
  HasVis: Boolean;
begin
  while FLex.TokenID in VisibilityTokens + ClassMemberTokens do
  begin
    HasVis := False;
    if FLex.TokenID in VisibilityTokens then
    begin
      MatchCreateLeafAndPush(FLex.TokenID);
      HasVis := True;
    end;

    try
      BuildClassMembers; // �� Visibility ��ѭ�� Build ���
    finally
      if HasVis then
        PopLeaf;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildClassMembers;
begin
  while FLex.TokenID in ClassMemberTokens do
  begin
    case FLex.TokenID of
      tkProperty:
        BuildClassProperty;
      tkProcedure, tkFunction, tkConstructor, tkDestructor, tkClass:
        BuildClassMethod;
      tkType:
        BuildClassTypeSection;
      tkConst:
        BuildClassConstSection;
      tkSquareOpen:
        BuildSingleAttribute;
    else
      BuildClassField;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildClassField;
begin
  repeat
    MatchCreateLeafAndPush(tkNone, cntClassField);

    try
      BuildIdentList;
      MatchCreateLeafAndStep(tkColon);
      BuildCommonType;
    finally
      PopLeaf;
    end;

    if FLex.TokenID = tkSemiColon then
      MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

    if FLex.TokenID <> tkIdentifier then
      Break;
  until False;
end;

procedure TCnPasAstGenerator.BuildConstSection;
begin
  if FLex.TokenID = tkConst then
    MarkReturnFlag(MatchCreateLeafAndPush(tkConst))
  else if FLex.TokenID = tkResourcestring then
    MarkReturnFlag(MatchCreateLeafAndPush(tkResourcestring));

  try
    while FLex.TokenID in [tkIdentifier, tkAmpersand] do
    begin
      BuildConstDecl;
      MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildConstDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntConstDecl);

  try
    BuildIdent;
    if FLex.TokenID = tkEqual then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildConstExpression;
    end
    else if FLex.TokenID = tkColon then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildCommonType;

      MatchCreateLeafAndStep(tkEqual);
      BuildTypedConstant;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildDirectives(NeedSemicolon: Boolean);
begin
  FReturnRef := nil;
  while FLex.TokenID in DirectiveTokens do
  begin
    BuildDirective;

    if NeedSemicolon and (FLex.TokenID = tkSemiColon) then
      FReturnRef := MatchCreateLeafAndStep(FLex.TokenID);
  end;

  if FReturnRef <> nil then
    FReturnRef.Return := True;
end;

procedure TCnPasAstGenerator.BuildConstExpressionInType;
begin
  MatchCreateLeafAndPush(tkNone, cntConstExpressionInType);

  try
    BuildSimpleExpression;
    while FLex.TokenID in RelOpTokens - [tkEqual, tkGreater, tkLower] do
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildSimpleExpression;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildLibrary;
begin

end;

procedure TCnPasAstGenerator.BuildProgram;
begin
  MatchCreateLeafAndPush(tkProgram);

  try
    BuildIdent;

    if FLex.TokenID = tkRoundOpen then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildIdentList;
      MatchCreateLeafAndStep(tkRoundClose);
    end;

    MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
    BuildProgramBlock;

    MatchCreateLeafAndStep(tkPoint);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildUnit;
begin
  MatchCreateLeafAndPush(tkUnit);

  try
    BuildIdent; // ��֧�ֵ�Ԫ���� platform ����

    MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));

    BuildInterfaceSection;

    BuildImplementationSection;

    if FLex.TokenID in [tkInitialization, tkBegin] then
      BuildInitSection;

    MatchCreateLeafAndStep(tkEnd);
    MarkReturnFlag(MatchCreateLeafAndStep(tkPoint));
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildProgramBlock;
begin
  while FLex.TokenID = tkUses do
    BuildUsesClause;

  while FLex.TokenID in DeclSectionTokens do
    BuildDeclSection;

  BuildCompoundStatement;
end;

procedure TCnPasAstGenerator.BuildImplementationSection;
begin
  MatchCreateLeafAndPush(tkImplementation);

  try
    while FLex.TokenID = tkUses do
      BuildUsesClause;

    while FLex.TokenID in DeclSectionTokens do
      BuildDeclSection;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildInterfaceSection;
begin
  MarkReturnFlag(MatchCreateLeafAndPush(tkInterface, cntInterfaceSection));

  try
    while FLex.TokenID = tkUses do
      BuildUsesClause;

    if FLex.TokenID in InterfaceDeclTokens then
      BuildInterfaceDecl;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildInitSection;
begin
  MatchCreateLeafAndPush(tkInitialization);

  try
    if FLex.TokenID <> tkFinalization then
      BuildStatementList;
  finally
    PopLeaf;
  end;

  if FLex.TokenID = tkFinalization then
  begin
    MatchCreateLeafAndPush(tkFinalization);

    try
      if FLex.TokenID <> tkEnd then
        BuildStatementList;
    finally
      PopLeaf;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildDeclSection;
begin
  while FLex.TokenID in DeclSectionTokens do
  begin
    case FLex.TokenID of
      tkLabel:
        BuildLabelDeclSection;
      tkConst, tkResourcestring:
        BuildConstSection;
      tkType:
        BuildTypeSection;
      tkVar, tkThreadvar:
        BuildVarSection;
      tkExports:
        BuildExportsSection;
      tkClass, tkProcedure, tkFunction, tkConstructor, tkDestructor:
        BuildProcedureDeclSection;
      tkSquareOpen:
        BuildSingleAttribute;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildInterfaceDecl;
begin
  while FLex.TokenID in InterfaceDeclTokens do
  begin
    case FLex.TokenID of
      tkConst, tkResourcestring:
        BuildConstSection;
      tkType:
        BuildTypeSection;
      tkVar, tkThreadvar:
        BuildVarSection;
      tkProcedure, tkFunction:
        BuildExportedHeading;
      tkExports:
        BuildExportsSection;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildCompoundStatement;
begin
  MatchCreateLeafAndPush(tkNone, cntCompoundStatement);

  try
    if FLex.TokenID = tkBegin then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildStatementList;
      MatchCreateLeafAndStep(tkEnd);
    end
    else if FLex.TokenID = tkAsm then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BulidAsmBlock;
      MatchCreateLeafAndStep(tkEnd);
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildStatementList;
begin
  while FLex.TokenID = tkSemiColon do
    MatchCreateLeafAndStep(FLex.TokenID);

  repeat
    while FLex.TokenID = tkSemiColon do
      MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

    BuildStatement;

    while FLex.TokenID = tkSemiColon do
      MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

    if not (FLex.TokenID in StatementTokens) then
      Break;
  until False;
end;

procedure TCnPasAstGenerator.BuildExportsList;
begin
  repeat
    BuildexportsDecl;
    if FLex.TokenID = tkComma then
      MatchCreateLeafAndStep(FLex.TokenID)
    else
      Break;
  until False;
end;

procedure TCnPasAstGenerator.BuildExportsSection;
begin
  MatchCreateLeafAndPush(tkExports);

  try
    BuildExportsList;
    MatchCreateLeafAndStep(tkSemiColon);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildExportsDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntExportDecl);

  try
    BuildIdent;
    if FLex.TokenID = tkRoundOpen then
      BuildFormalParameters;

    if FLex.TokenID = tkColon then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildSimpleType;
    end;

    BuildDirectives(False); // Export �����ﲻҪ����ֺţ�ԭ��̫��ȷ
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSimpleType;
begin
  if FLex.TokenID = tkRoundOpen then
    BuildSubrangeType
  else
  begin
    BuildConstExpressionInType;
    if FLex.TokenID = tkDotdot then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildConstExpressionInType;
    end;
  end;
end;

procedure TCnPasAstGenerator.BuildExportedHeading;
begin
  if FLex.TokenID = tkProcedure then
    BuildProcedureHeading
  else if FLex.TokenID = tkFunction then
    BuildFunctionHeading;

  if FLex.TokenID = tkSemiColon then
    MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

  BuildDirectives;
end;

procedure TCnPasAstGenerator.BuildDirective;
var
  CanExpr: Boolean;
begin
  if FLex.TokenID in DirectiveTokens then
  begin
    CanExpr := FLex.TokenID in DirectiveTokensWithExpressions;
    MatchCreateLeafAndStep(FLex.TokenID);

    if CanExpr and not (FLex.TokenID in DirectiveTokens + [tkSemiColon]) then
      BuildConstExpression;
  end;
end;

procedure TCnPasAstGenerator.BuildRecordConstant;
begin
  MatchCreateLeafAndPush(tkRoundOpen, cntRecordConstant);

  try
    repeat
      BuildRecordFieldConstant;
      if FLex.TokenID = tkSemiColon then  // ĩβ�ķֺſ�Ҫ�ɲ�Ҫ�����ܰ�û�ֺ���Ϊ�����ı�ǣ�ֻ����������
        MatchCreateLeafAndStep(FLex.TokenID);

      if FLex.TokenID = tkRoundClose then
        Break;
    until False;
  finally
    PopLeaf;
  end;
  MatchCreateLeafAndStep(tkRoundClose);
end;

procedure TCnPasAstGenerator.BuildArrayConstant;
begin
  MatchCreateLeafAndPush(tkRoundOpen, cntArrayConstant);

  try
    repeat
      BuildTypedConstant;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(FLex.TokenID)
      else
        Break;
    until False;
  finally
    PopLeaf;
  end;
  MatchCreateLeafAndStep(tkRoundClose);
end;

procedure TCnPasAstGenerator.BuildRecordFieldConstant;
begin
  MatchCreateLeafAndPush(tkNone, cntRecordFieldConstant);

  try
    BuildIdent;
    MatchCreateLeafAndStep(tkColon);
    BuildTypedConstant;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildStatement;
begin
  MatchCreateLeafAndPush(tkNone, cntStatememt);

  try
    if ForwardToken() = tkColon then
    begin
      BuildLabelId;
      MatchCreateLeafAndStep(tkColon);
    end;

    if FLex.TokenID in SimpleStatementTokens then
      BuildSimpleStatement
    else if FLex.TokenID in StructStatementTokens then
      BuildStructStatement;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildStructStatement;
begin
  case FLex.TokenID of
    tkBegin, tkAsm:  BuildCompoundStatement;
    tkIf:     BuildIfStatement;
    tkCase:   BuildCaseStatement;
    tkRepeat: BuildRepeatStatement;
    tkWhile:  BuildWhileStatement;
    tkFor:    BuildForStatement;
    tkWith:   BuildWithStatement;
    tkTry:    BuildTryStatement;
    tkRaise:  BuildRaiseStatement;
  end;
end;

procedure TCnPasAstGenerator.BuildCaseStatement;
begin
  MatchCreateLeafAndPush(tkCase);

  try
    BuildExpression;
    MatchCreateLeafAndStep(tkOf);

    repeat
      BuildCaseSelector;

      if FLex.TokenID = tkSemiColon then
        MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

      if FLex.TokenID in [tkElse, tkEnd] then
        Break;
    until False;

    if FLex.TokenID = tkElse then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      if FLex.TokenID <> tkEnd then
        BuildStatementList;
    end;

    if FLex.TokenID = tkSemiColon then
      MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

    MatchCreateLeafAndStep(tkEnd);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildForStatement;
begin
  MatchCreateLeafAndPush(tkFor);

  try
    BuildQualId; // ��֧���ֳ����� var

    if FLex.TokenID = tkAssign then
    begin
      MatchCreateLeafAndStep(tkAssign);
      BuildExpression;

      if FLex.TokenID in [tkTo, tkDownto] then
      begin
        MatchCreateLeafAndStep(FLex.TokenID);
        BuildExpression;
        MatchCreateLeafAndStep(tkDo);
        BuildStatement;
      end;
    end
    else if FLex.TokenID = tkIn then
    begin
      BuildExpression;
      MatchCreateLeafAndStep(tkDo);
      BuildStatement;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildIfStatement;
begin
  MatchCreateLeafAndPush(tkIf);

  try
    BuildExpression;
    MatchCreateLeafAndStep(tkThen);
    BuildStatement;

    if FLex.TokenID = tkElse then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      if FLex.TokenID = tkIf then
        BuildIfStatement
      else
        BuildStatement;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildRaiseStatement;
begin
  MatchCreateLeafAndPush(tkRaise);

  try
    BuildExpression;

    if FLex.TokenID = tkAt then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildExpression;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildRepeatStatement;
begin
  MatchCreateLeafAndPush(tkRepeat);

  try
    BuildStatementList;
    MatchCreateLeafAndStep(tkUntil);
    BuildExpression;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildTryStatement;
begin
  MatchCreateLeafAndPush(tkTry);

  try
    BuildStatementList;
    if not (FLex.TokenID in [tkExcept, tkFinally]) then
      BuildStatementList;

    if FLex.TokenID = tkFinally then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildStatementList;
      MatchCreateLeafAndStep(tkEnd);
    end
    else if FLex.TokenID = tkExcept then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      if FLex.TokenID <> tkEnd then
      begin
        if FLex.TokenID in [tkOn, tkElse] then
        begin
          while FLex.TokenID = tkOn do
            BuildExceptionHandler;

          if FLex.TokenID = tkElse then
            BuildStatementList;

          if FLex.TokenID = tkSemiColon then
            MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
        end
        else
          BuildStatementList;
      end;

      MatchCreateLeafAndStep(tkEnd);
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildWhileStatement;
begin
  MatchCreateLeafAndPush(tkWhile);

  try
    BuildExpression;
    MatchCreateLeafAndStep(tkDo);
    BuildStatement;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildWithStatement;
begin
  MatchCreateLeafAndPush(tkWith);

  try
    BuildExpressionList;
    MatchCreateLeafAndStep(tkDo);
    BuildStatement;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildExceptionHandler;
begin
  MatchCreateLeafAndPush(tkOn);

  try
    BuildIdent;
    if FLex.TokenID = tkColon then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildIdent;
    end;

    MatchCreateLeafAndStep(tkDo);
    BuildStatement;

    if FLex.TokenID = tkSemiColon then
      MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildCaseSelector;
begin
  MatchCreateLeafAndPush(tkNone, cntCaseSelector);

  try
    repeat
      BuildCaseLabel;
      if FLex.TokenID = tkComma then
        MatchCreateLeafAndStep(FLex.TokenID)
      else
        Break;
    until False;

    MarkReturnFlag(MatchCreateLeafAndStep(tkColon));
    BuildStatement;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildCaseLabel;
begin
  MatchCreateLeafAndPush(tkNone, cntCaseLabel);

  try
    BuildConstExpression;
    if FLex.TokenID = tkDotdot then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildConstExpression;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildLabelDeclSection;
begin
  MatchCreateLeafAndPush(tkLabel);

  try
    repeat
      BuildLabelId;
      if FLex.TokenID <> tkComma then
        Break;
    until False;

    MarkReturnFlag(MatchCreateLeafAndStep(tkSemiColon));
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildProcedureDeclSection;
begin
  if FLex.TokenID = tkClass then
    MatchCreateLeafAndStep(FLex.TokenID);

  case FLex.TokenID of
    tkProcedure, tkConstructor, tkDestructor, tkFunction:
      BuildProcedureFunctionDecl;
  end;
end;

procedure TCnPasAstGenerator.BuildProcedureFunctionDecl;
var
  IsExternal: Boolean;
  IsForward: Boolean;
begin
  MatchCreateLeafAndPush(tkNone, cntProcedureFunctionDecl);

  try
    if FLex.TokenID = tkFunction then
      BuildFunctionHeading
    else
      BuildProcedureHeading;

    if FLex.TokenID = tkSemiColon then
      MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));

    IsExternal := False;
    IsForward := False;

    while FLex.TokenID in DirectiveTokens do
    begin
      if FLex.TokenID = tkExternal then
        IsExternal := True
      else if FLex.TokenID = tkForward then
        IsForward := True;

      BuildDirective;
      if FLex.TokenID = tkSemiColon then
        MatchCreateLeafAndStep(FLex.TokenID);
    end;

    if ((not IsExternal)  and (not IsForward)) and
       (FLex.TokenID in [tkBegin, tkAsm] + DeclSectionTokens) then
    begin
      BuildBlock;
      if FLex.TokenID = tkSemicolon then
        MarkReturnFlag(MatchCreateLeafAndStep(FLex.TokenID));
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildBlock;
begin
  while FLex.TokenID in DeclSectionTokens do
    BuildDeclSection;

  BuildCompoundStatement;
end;

procedure TCnPasAstGenerator.BuildClassConstantDecl;
begin
  MatchCreateLeafAndPush(tkNone, cntClassConstantDecl);

  try
    BuildIdent;

    if FLex.TokenID = tkEqual then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildConstExpression;
    end
    else if FLex.TokenID = tkColon then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildCommonType;

      MatchCreateLeafAndStep(tkEqual);
      BuildTypedConstant;
    end;

    BuildDirectives(False);
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.SkipComments;
begin
  while FLex.TokenID in SpaceTokens + CommentTokens do
  begin
    if FLex.TokenID in CommentTokens then
      MatchCreateLeaf(FLex.TokenID); // ��ͷû�� NextToken

    FLex.Next;
  end;
end;

procedure TCnPasAstGenerator.BulidAsmBlock;
var
  T: TCnPasAstLeaf;
begin
  if FLex.TokenID = tkEnd then
    Exit;

  T := MatchCreateLeafAndPush(tkNone, cntAsmBlock);
  try
    while FLex.TokenID <> tkEnd do
    begin
      if T <> nil then
        T.Text := T.Text + FLex.Token;
      FLex.Next;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildSingleAttribute;
begin
  MatchCreateLeafAndPush(tkNone, cntSingleAttribute);
  // Pop ֮ǰ�ڲ����ݾ�Ϊ�ó��� SingleAttribute ֮��

  try
    if FLex.TokenID = tkSquareOpen then
    begin
      MatchCreateLeafAndPush(tkSquareOpen);
      try
        repeat
          BuildAttributeItem;
          if FLex.TokenID = tkComma then
            MatchCreateLeafAndStep(tkComma)
          else
            Break;
        until False;
      finally
        PopLeaf;
      end;
      MatchCreateLeafAndStep(tkSquareClose);
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.BuildAttributeItem;
begin
  MatchCreateLeafAndPush(tkNone, cntAttributeItem);
  // Pop ֮ǰ�ڲ����ݾ�Ϊ�ó��� AttributeItem ֮��

  try
    BuildIdent;

    if FLex.TokenID = tkRoundOpen then
    begin
      MatchCreateLeafAndPush(tkRoundOpen);
      try
        BuildExpressionList;
      finally
        PopLeaf;
      end;
      MatchCreateLeafAndStep(tkRoundClose);
    end
    else if FLex.TokenID in [tkColon, tkEqual] then
    begin
      MatchCreateLeafAndStep(FLex.TokenID);
      BuildIdent;
    end;
  finally
    PopLeaf;
  end;
end;

procedure TCnPasAstGenerator.MarkReturnFlag(ALeaf: TCnPasAstLeaf);
begin
  if ALeaf <> nil then
    ALeaf.Return := True;
end;

procedure TCnPasAstGenerator.MarkNoSpaceBehindFlag(ALeaf: TCnPasAstLeaf);
begin
  if ALeaf <> nil then
    ALeaf.NoSpaceBehind := True;
end;

procedure TCnPasAstGenerator.MarkNoSpaceBeforeFlag(ALeaf: TCnPasAstLeaf);
begin
  if ALeaf <> nil then
    ALeaf.NoSpaceBefore := True;
end;

{ TCnPasAstTree }

function TCnPasAstTree.ConvertToCppCode: string;
begin
  // �� implementation �������� cpp �ļ�
end;

function TCnPasAstTree.ConvertToHppCode: string;
begin
  // �� interface �������� h �ļ�
end;

function TCnPasAstTree.GetItems(AbsoluteIndex: Integer): TCnPasAstLeaf;
begin
  Result := TCnPasAstLeaf(inherited GetItems(AbsoluteIndex));
end;

function TCnPasAstTree.GetRoot: TCnPasAstLeaf;
begin
  Result := TCnPasAstLeaf(inherited GetRoot);
end;

function TCnPasAstTree.ReConstructPascalCode: string;
begin
  Result := (FRoot as TCnPasAstLeaf).GetPascalCode;
end;

{ TCnPasAstLeaf }

function TCnPasAstLeaf.ConvertNumber: string;
begin
  Result := Text;
end;

function TCnPasAstLeaf.ConvertQualId: string;
begin

end;

function TCnPasAstLeaf.ConvertString: string;
var
  P: PChar;
  I: Integer;
  SB: TCnStringBuilder;
begin
  // ɨ���ڲ������ź�#�ȣ�ת���� C �����ַ������
  P := @Text[1];
  I := 0;
  SB := TCnStringBuilder.Create;

  try
    while P[I] <> #0 do
    begin
      case P[I] of
        '''':
          begin
            // ������
            if P[I + 1] = '''' then // �������������Ŵ���һ��������
            begin
              SB.Append('''');
              Inc(I);
            end;
          end;
        '#':
          begin
            // # ��
            SB.Append('\');
            if P[I + 1] = '$' then
            begin
              SB.Append('0x');
              Inc(I);
            end;
          end;
        '"':  // ˫���ţ�C �ַ�����Ҫת��
          begin
            SB.Append('\');
            SB.Append('"');
          end;
      else
        SB.Append(P[I]);
      end;
      Inc(I);
    end;

    SB.Append('"');
    Result := '"' + SB.ToString;
  finally
    SB.Free;
  end;
end;

function TCnPasAstLeaf.GetCppCode: string;
begin
  case FTokenKind of // �����������ͺͻ����ؼ���
    tkString, tkAsciiChar:
      begin
        Result := ConvertString;
      end;
    tkNumber, tkFloat:
      begin
        Result := ConvertNumber;
      end;
    tkPlus, tkMinus, tkStar, tkSlash, tkRoundOpen, tkRoundClose, tkSquareOpen, tkSquareClose, tkPoint:
      Result := Text; // ����������ţ�С���š������Ų���
    tkGreater, tkGreaterEqual, tkLower, tkLowerEqual:
      Result := Text; // ���ĸ��ȽϷ��Ų���
    tkNotEqual:
      Result := '!=';  // ������
    tkEqual:
      Result := '==';
    tkDiv:
      Result := '\';
    tkMod:
      Result := '%';
    tkShl:
      Result := '<<';
    tkShr:
      Result := '>>';
    tkAssign:
      Result := '=';
    tkAnd:
      begin

      end;
    tkOr:
      begin

      end;
    tkNot:
      begin

      end;
    tkXor:
      Result := '^';
    tkNil:
      Result := 'NULL';
  end;

  if Result <> '' then
    Exit;

  case FNodeType of
    cntQualId:
      begin
        Result := ConvertQualId;
      end;
  end;
end;

function TCnPasAstLeaf.GetItems(AIndex: Integer): TCnPasAstLeaf;
begin
  Result := TCnPasAstLeaf(inherited GetItems(AIndex));
end;

function TCnPasAstLeaf.GetParent: TCnPasAstLeaf;
begin
  Result := TCnPasAstLeaf(inherited GetParent);
end;

function TCnPasAstLeaf.GetPascalCode: string;
var
  I: Integer;
  S: string;
  Prev, Son: TTokenKind;
  NSP: Boolean;
begin
  if FReturn or (FTokenKind in [tkBorComment, tkAnsiComment, tkSlashesComment, // ע�Ͷ������Ȼ���
    tkBegin, tkThen, tkDo, tkRepeat,                                           // ��Щ���涼����
    tkExcept, tkExports, tkFinally, tkInitialization, tkFinalization, tkAsm,
    tkImplementation, tkRecord, tkPrivate, tkProtected, tkPublic, tkPublished]) then
    Result := Text + #13#10
//  else if FTokenKind = tkCRLFCo then
//    Result := ''
  else
    Result := Text;

  for I := 0 to Count - 1 do
  begin
    Son := Items[I].TokenKind;
    S := Items[I].GetPascalCode;
    if Result = '' then
      Result := S
    else if S <> '' then
    begin
      if I = 0 then
        Prev := FTokenKind
      else
        Prev := Items[I - 1].TokenKind;

      NSP := FNoSpaceBehind or Items[I].NoSpaceBefore or    // ������ڵ���治Ҫ�ո񣬻�ǰ�ӽڵ�ǰ�治Ҫ�ո�
        (Prev in [tkRoundOpen, tkSquareOpen, tkPoint, tkDotDot]) or            // ǰһ���ڵ��������Щ����ǰһ���ڵ���治Ҫ�ո�
        (Son in [tkPoint, tkDotdot, tkPointerSymbol, tkSemiColon, tkColon, // ��ǰ�ӽڵ��������Щ����ǰ�ӽڵ�ǰ�治Ҫ�ո�
        tkRoundClose, tkSquareClose, tkComma]);

      if not NSP then
      begin
        // ����Щ FuncCall(  class(  array[ �Ĳ�Ҫ�ո�û����������������⴦��
        if ((Prev in [tkClass, tkIdentifier]) and (Son in [tkRoundOpen]))
          or ((Prev in [tkArray]) and (Son in [tkSquareOpen])) then
          NSP := True;
      end;

      if NSP then
        Result := Result + S
      else
        Result := Result + ' ' + S;
    end;
  end;
end;

procedure TCnPasAstLeaf.SetItems(AIndex: Integer;
  const Value: TCnPasAstLeaf);
begin
  inherited SetItems(AIndex, Value);
end;

end.
