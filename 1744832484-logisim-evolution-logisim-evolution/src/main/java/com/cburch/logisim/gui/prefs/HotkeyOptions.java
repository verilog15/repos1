/*
 * Logisim-evolution - digital logic design tool and simulator
 * Copyright by the Logisim-evolution developers
 *
 * https://github.com/logisim-evolution/
 *
 * This is free software released under GNU GPLv3 license
 */

package com.cburch.logisim.gui.prefs;

import static com.cburch.logisim.gui.Strings.S;

import com.cburch.logisim.prefs.AppPreferences;
import com.cburch.logisim.prefs.PrefMonitor;
import com.cburch.logisim.prefs.PrefMonitorKeyStroke;
import com.cburch.logisim.util.JAdjustableScroll;
import com.cburch.logisim.util.JHotkeyInput;
import com.cburch.logisim.util.TableLayout;
import java.awt.Dimension;
import java.awt.event.InputEvent;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.KeyStroke;
import javax.swing.Timer;

class HotkeyOptions extends OptionsPanel {
  private static final long serialVersionUID = 1L;
  /*
   * Hotkey Options TAB
   *
   * Author: Hanyuan Zhao <2524395907@qq.com>
   *
   * Description:
   * This is the hotkey settings Tab in the preferences.
   * Allowing users to decide which hotkey to bind to the specific function.
   *
   * To add your own hotkey bindings from your code, you need some operations as follows.
   * Firstly add your hotkey configurations to AppPreferences and set up their strings in resources
   * Fill the resetHotkeys method in AppPreferences, adding the reset code for your hotkeys
   * Set up the hotkey in your code by accessing AppPreferences.HOTKEY_ADD_BY_YOU
   * Do not forget to sync with the user's settings.
   * You should go modifying hotkeySync in AppPreferences, adding your codes there.
   *
   * Now the hotkey options don't involve all the bindings in logisim.
   * The hotkeys chosen by the user might have conflict with
   * some build-in key bindings until all key bindings can be set in this tab.
   * TODO: If you are available, you can bind them in order to make logisim feel better
   *
   * */
  protected static List<PrefMonitor<KeyStroke>> hotkeys = new ArrayList<>();
  private final List<JHotkeyInput> keyInputList;
  private final List<JLabel> keyLabels;
  private final JLabel menuKeyHeaderLabel;
  private final JLabel normalKeyHeaderLabel;
  private JHotkeyInput northBtn;
  private JHotkeyInput southBtn;
  private JHotkeyInput eastBtn;
  private JHotkeyInput westBtn;
  private final JButton resetBtn;
  private final JLabel orientDescLabel;
  private final JLabel orientNorthLabel;
  private final JLabel orientEastLabel;
  private final JLabel orientSouthLabel;
  private final JLabel orientWestLabel;
  private boolean preferredWidthSet = false;

  public HotkeyOptions(PreferencesFrame window) {
    super(window);
    this.setLayout(new TableLayout(1));

    /* settings the layout up */
    resetBtn = new JButton();
    resetBtn.addActionListener(e -> AppPreferences.resetHotkeys());
    add(resetBtn);
    add(new JLabel(" "));

    menuKeyHeaderLabel = new JLabel();
    add(menuKeyHeaderLabel);
    add(new JLabel(" "));
    JPanel menuKeyPanel = new JPanel();
    JAdjustableScroll menuKeyScrollPane = new JAdjustableScroll(menuKeyPanel);
    add(menuKeyScrollPane);

    add(new JLabel(" "));
    normalKeyHeaderLabel = new JLabel();
    add(normalKeyHeaderLabel);
    add(new JLabel(" "));
    JPanel normalKeyPanel = new JPanel();
    JAdjustableScroll normalKeyScrollPane = new JAdjustableScroll(normalKeyPanel);
    add(normalKeyScrollPane);

    menuKeyPanel.setMaximumSize(new Dimension(400, 400));
    menuKeyPanel.setLayout(new TableLayout(2));
    normalKeyPanel.setMaximumSize(new Dimension(400, 400));
    normalKeyPanel.setLayout(new TableLayout(2));

    /* bind up the hotkeys */
    Field[] fields = AppPreferences.class.getDeclaredFields();
    try {
      for (var f : fields) {
        String name = f.getName();
        if (name.contains("HOTKEY_")) {
          @SuppressWarnings("unchecked")
          PrefMonitor<KeyStroke> keyStroke = (PrefMonitor<KeyStroke>) f.get(AppPreferences.class);
          hotkeys.add(keyStroke);
        }
      }
    } catch (Exception e) {
      AppPreferences.hotkeyReflectError(e);
    }

    /* initialize the hotkey labels and hotkey-inputs */
    keyInputList = new ArrayList<>();
    keyLabels = new ArrayList<>();
    for (int i = 0; i < hotkeys.size(); i++) {
      keyInputList.add(new JHotkeyInput(window, ""));
      keyLabels.add(new JLabel());
    }
    for (int i = 0; i < hotkeys.size(); i++) {
      /* I do this chore because they have a different layout */
      var prefKeyStroke = ((PrefMonitorKeyStroke) hotkeys.get(i));
      if (hotkeys.get(i) == AppPreferences.HOTKEY_DIR_NORTH
          || hotkeys.get(i) == AppPreferences.HOTKEY_DIR_SOUTH
          || hotkeys.get(i) == AppPreferences.HOTKEY_DIR_EAST
          || hotkeys.get(i) == AppPreferences.HOTKEY_DIR_WEST) {
        if (hotkeys.get(i) == AppPreferences.HOTKEY_DIR_NORTH) {
          northBtn = new JHotkeyInput(window, prefKeyStroke.getDisplayString());
          keyInputList.set(i, northBtn);
        }
        if (hotkeys.get(i) == AppPreferences.HOTKEY_DIR_SOUTH) {
          southBtn = new JHotkeyInput(window, prefKeyStroke.getDisplayString());
          keyInputList.set(i, southBtn);
        }
        if (hotkeys.get(i) == AppPreferences.HOTKEY_DIR_EAST) {
          eastBtn = new JHotkeyInput(window, prefKeyStroke.getDisplayString());
          keyInputList.set(i, eastBtn);
        }
        if (hotkeys.get(i) == AppPreferences.HOTKEY_DIR_WEST) {
          westBtn = new JHotkeyInput(window, prefKeyStroke.getDisplayString());
          keyInputList.set(i, westBtn);
        }
        keyInputList.get(i).setEnabled(prefKeyStroke.canModify());
        keyInputList.get(i).setBoundKeyStroke(prefKeyStroke);
        continue;
      }
      keyLabels.get(i).setText(S.get(prefKeyStroke.getName()) + "  ");
      keyInputList.set(i, new JHotkeyInput(window, prefKeyStroke.getDisplayString()));
      keyInputList.get(i).setEnabled(prefKeyStroke.canModify());
      keyInputList.get(i).setBoundKeyStroke(prefKeyStroke);
      if (prefKeyStroke.needMetaKey()) {
        menuKeyPanel.add(keyLabels.get(i));
        menuKeyPanel.add(keyInputList.get(i));
      } else {
        normalKeyPanel.add(keyLabels.get(i));
        normalKeyPanel.add(keyInputList.get(i));
      }
    }

    /* adding layout for arrow hotkeys */
    normalKeyPanel.add(new JLabel(" "));
    normalKeyPanel.add(new JLabel(" "));

    orientDescLabel = new JLabel();
    normalKeyPanel.add(orientDescLabel);
    normalKeyPanel.add(new JLabel(" "));
    JPanel panelLeft = new JPanel();
    JPanel panelRight = new JPanel();
    panelLeft.setLayout(new TableLayout(3));
    panelRight.setLayout(new TableLayout(3));

    orientEastLabel = new JLabel();
    orientNorthLabel = new JLabel();
    orientSouthLabel = new JLabel();
    orientWestLabel = new JLabel();
    panelLeft.add(new JLabel(" "));
    panelLeft.add(orientNorthLabel);
    panelLeft.add(new JLabel(" "));
    panelLeft.add(orientWestLabel);
    panelLeft.add(orientSouthLabel);
    panelLeft.add(orientEastLabel);
    normalKeyPanel.add(panelLeft);
    panelRight.add(new JLabel(" "));
    panelRight.add(northBtn);
    panelRight.add(new JLabel(" "));
    panelRight.add(westBtn);
    panelRight.add(southBtn);
    panelRight.add(eastBtn);
    normalKeyPanel.add(panelRight);

    var that = this;
    /* this timer deals with the preferred width and the theme changing problem */
    new Timer(200, e -> {
      int menuWidth = menuKeyPanel.getWidth();
      int normalWidth = normalKeyPanel.getWidth();
      if (normalWidth > 0 && normalWidth < that.getWidth() * 0.8 && !preferredWidthSet) {
        menuKeyScrollPane.preferredWidth = menuWidth + 40;
        menuKeyPanel.setPreferredSize(new Dimension(menuKeyPanel.getSize().width,
            menuKeyPanel.getPreferredSize().height));
        normalKeyScrollPane.preferredWidth = normalWidth + 40;
        normalKeyPanel.setPreferredSize(new Dimension(normalKeyPanel.getSize().width,
            normalKeyPanel.getPreferredSize().height));
        preferredWidthSet = true;
      }
    }).start();

    AppPreferences.getPrefs().addPreferenceChangeListener(evt -> {
      AppPreferences.hotkeySync();
      for (int i = 0; i < hotkeys.size(); i++) {
        keyInputList.get(i).resetText(((PrefMonitorKeyStroke) hotkeys.get(i)).getDisplayString());
      }
      for (var h : keyInputList) {
        h.exitEditMode();
      }
    });
  }

  @Override
  public String getHelpText() {
    return S.get("hotkeyOptHelp");
  }

  @Override
  public String getTitle() {
    return S.get("hotkeyOptTitle");
  }

  @Override
  public void localeChanged() {
    menuKeyHeaderLabel.setText(S.get("hotkeyOptMenuKeyHeader",
        InputEvent.getModifiersExText(AppPreferences.hotkeyMenuMask)));
    normalKeyHeaderLabel.setText(S.get("hotkeyOptNormalKeyHeader",
        InputEvent.getModifiersExText(AppPreferences.hotkeyMenuMask)));
    resetBtn.setText(S.get("hotkeyOptResetBtn"));
    orientDescLabel.setText(S.get("hotkeyOptOrientDesc"));
    orientEastLabel.setText(" " + S.get("hotkeyDirEast") + " ");
    orientWestLabel.setText(" " + S.get("hotkeyDirWest") + " ");
    orientSouthLabel.setText(" " + S.get("hotkeyDirSouth") + " ");
    orientNorthLabel.setText(" " + S.get("hotkeyDirNorth") + " ");
    for (int i = 0; i < hotkeys.size(); i++) {
      var prefKeyStroke = ((PrefMonitorKeyStroke) hotkeys.get(i));
      if (hotkeys.get(i) == AppPreferences.HOTKEY_DIR_NORTH
          || hotkeys.get(i) == AppPreferences.HOTKEY_DIR_SOUTH
          || hotkeys.get(i) == AppPreferences.HOTKEY_DIR_EAST
          || hotkeys.get(i) == AppPreferences.HOTKEY_DIR_WEST) {
        continue;
      }
      keyLabels.get(i).setText(S.get(prefKeyStroke.getName()) + "  ");
    }
  }
}
