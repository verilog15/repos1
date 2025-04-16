object CnAICoderChatForm: TCnAICoderChatForm
  Left = 700
  Top = 110
  Width = 549
  Height = 704
  Caption = 'AI Coder Chat'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  Position = poScreenCenter
  OnCreate = FormCreate
  OnDestroy = FormDestroy
  PixelsPerInch = 96
  TextHeight = 13
  object spl1: TSplitter
    Left = 0
    Top = 567
    Width = 541
    Height = 3
    Cursor = crVSplit
    Align = alBottom
  end
  object pnlChat: TPanel
    Left = 0
    Top = 30
    Width = 541
    Height = 537
    Align = alClient
    BevelInner = bvLowered
    BevelOuter = bvNone
    TabOrder = 0
  end
  object tlbAICoder: TToolBar
    Left = 0
    Top = 0
    Width = 541
    Height = 30
    BorderWidth = 1
    EdgeBorders = [ebLeft, ebTop, ebRight, ebBottom]
    Flat = True
    Images = dmCnSharedImages.Images
    ParentShowHint = False
    ShowHint = True
    TabOrder = 1
    object btnToggleSend: TToolButton
      Left = 0
      Top = 0
      Action = actToggleSend
    end
    object btnClear: TToolButton
      Left = 23
      Top = 0
      Action = actClear
    end
    object btn1: TToolButton
      Left = 46
      Top = 0
      Width = 8
      ImageIndex = 2
      Style = tbsSeparator
    end
    object btnReferSelection: TToolButton
      Left = 54
      Top = 0
      Hint = 'Attach Selected Code in Editor when Asking'
      ImageIndex = 88
      OnClick = btnReferSelectionClick
    end
    object btn3: TToolButton
      Left = 77
      Top = 0
      Width = 8
      Caption = 'btn3'
      ImageIndex = 3
      Style = tbsSeparator
    end
    object btnFont: TToolButton
      Left = 85
      Top = 0
      Action = actFont
    end
    object btnOption: TToolButton
      Left = 108
      Top = 0
      Action = actOption
    end
    object btnHelp: TToolButton
      Left = 131
      Top = 0
      Action = actHelp
    end
    object btn2: TToolButton
      Left = 154
      Top = 0
      Width = 8
      Caption = 'btn2'
      ImageIndex = 2
      Style = tbsSeparator
    end
    object cbbActiveEngine: TComboBox
      Left = 162
      Top = 0
      Width = 126
      Height = 21
      Style = csDropDownList
      Anchors = [akLeft, akTop, akRight]
      ItemHeight = 13
      TabOrder = 0
      OnChange = cbbActiveEngineChange
    end
  end
  object pnlTop: TPanel
    Left = 0
    Top = 570
    Width = 541
    Height = 107
    Align = alBottom
    BevelOuter = bvNone
    TabOrder = 2
    object btnMsgSend: TSpeedButton
      Left = 476
      Top = 16
      Width = 65
      Height = 65
      Hint = 'Enter to Send, Ctrl+Enter to Make a New Line'
      Anchors = [akTop, akRight]
      Flat = True
      Glyph.Data = {
        76080000424D7608000000000000760000002800000040000000400000000100
        0400000000000008000000000000000000001000000000000000000000000000
        8000008000000080800080000000800080008080000080808000C0C0C0000000
        FF0000FF000000FFFF00FF000000FF00FF00FFFF0000FFFFFF00FFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        8778FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF8
        222268FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF82
        22222278FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222268FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222222278FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222222222222222222268FFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222222222222226FFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222222222222222FFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222222222222222278FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222222222222268FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222222222278FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222222222268FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        2222222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        22222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF72
        222222228FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF87
        2222227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        22227FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        F878FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF}
      ParentShowHint = False
      ShowHint = True
      OnClick = btnMsgSendClick
    end
    object mmoSelf: TMemo
      Left = 0
      Top = 0
      Width = 474
      Height = 107
      Align = alLeft
      Anchors = [akLeft, akTop, akRight, akBottom]
      TabOrder = 0
      OnKeyPress = mmoSelfKeyPress
    end
  end
  object actlstAICoder: TActionList
    Images = dmCnSharedImages.Images
    OnUpdate = actlstAICoderUpdate
    Left = 40
    Top = 497
    object actToggleSend: TAction
      Caption = '&Toggle Send Area'
      Checked = True
      Hint = 'Toggle Send Area'
      ImageIndex = 34
      OnExecute = actToggleSendExecute
    end
    object actCopy: TAction
      Caption = '&Copy'
      Hint = 'Copy Chat Content'
      ImageIndex = 10
      OnExecute = actCopyExecute
    end
    object actCopyCode: TAction
      Caption = 'Copy Co&de'
      Hint = 'Copy Code Content between ``` and ```'
      ImageIndex = 56
      OnExecute = actCopyCodeExecute
    end
    object actClear: TAction
      Caption = 'C&lear'
      Hint = 'Clear Messages'
      ImageIndex = 13
      OnExecute = actClearExecute
    end
    object actFont: TAction
      Caption = '&Font'
      Hint = 'Change Font'
      ImageIndex = 29
      OnExecute = actFontExecute
    end
    object actOption: TAction
      Caption = '&Options...'
      Hint = 'Display Options'
      ImageIndex = 2
      OnExecute = actOptionExecute
    end
    object actHelp: TAction
      Caption = '&Help'
      Hint = 'Display Help'
      ImageIndex = 1
      OnExecute = actHelpExecute
    end
  end
  object pmChat: TPopupMenu
    Images = dmCnSharedImages.Images
    OnPopup = pmChatPopup
    Left = 112
    Top = 500
    object N1: TMenuItem
      Action = actCopy
    end
    object M1: TMenuItem
      Action = actCopyCode
    end
    object N2: TMenuItem
      Caption = '-'
    end
    object Clear1: TMenuItem
      Action = actClear
    end
  end
  object dlgFont: TFontDialog
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    MinFontSize = 0
    MaxFontSize = 0
    Left = 56
    Top = 594
  end
end
