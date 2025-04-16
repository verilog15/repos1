/*
 * Logisim-evolution - digital logic design tool and simulator
 * Copyright by the Logisim-evolution developers
 *
 * https://github.com/logisim-evolution/
 *
 * This is free software released under GNU GPLv3 license
 */

package com.cburch.logisim.gui.main;

import com.cburch.logisim.circuit.CircuitAttributes;
import com.cburch.logisim.circuit.CircuitEvent;
import com.cburch.logisim.circuit.CircuitListener;
import com.cburch.logisim.circuit.CircuitState;
import com.cburch.logisim.circuit.SubcircuitFactory;
import com.cburch.logisim.comp.Component;
import com.cburch.logisim.comp.ComponentFactory;
import com.cburch.logisim.data.AttributeEvent;
import com.cburch.logisim.data.AttributeListener;
import com.cburch.logisim.instance.StdAttr;
import java.util.ArrayList;
import java.util.Comparator;
import javax.swing.tree.TreeNode;

class SimulationTreeCircuitNode extends SimulationTreeNode
    implements CircuitListener, AttributeListener, Comparator<Component> {
  private final CircuitState circuitState;
  private final Component subcircComp;

  public SimulationTreeCircuitNode(
      SimulationTreeModel model,
      SimulationTreeCircuitNode parent,
      CircuitState circuitState,
      Component subcircComp) {
    super(model, parent);
    this.circuitState = circuitState;
    this.subcircComp = subcircComp;
    circuitState.getCircuit().addCircuitListener(this);
    if (subcircComp != null) {
      subcircComp.getAttributeSet().addAttributeListener(this);
    } else {
      circuitState.getCircuit().getStaticAttributes().addAttributeListener(this);
    }
    computeChildren();
  }

  //
  // AttributeListener methods
  @Override
  public void attributeValueChanged(AttributeEvent e) {
    Object attr = e.getAttribute();
    if (attr == CircuitAttributes.CIRCUIT_LABEL_ATTR || attr == StdAttr.LABEL) {
      model.fireNodeChanged(this);
    }
  }

  @Override
  public void circuitChanged(CircuitEvent event) {
    final var action = event.getAction();
    if (action == CircuitEvent.ACTION_SET_NAME) {
      model.fireNodeChanged(this);
    } else {
      if (computeChildren()) {
        model.fireStructureChanged(this);
      }
    }
  }

  @Override
  public int compare(Component a, Component b) {
    if (a != b) {
      final var nameA = a.getFactory().getDisplayName();
      final var nameB = b.getFactory().getDisplayName();
      final var ret = nameA.compareToIgnoreCase(nameB);
      if (ret != 0) return ret;
    }
    return a.getLocation().toString().compareTo(b.getLocation().toString());
  }

  // returns true if changed
  private boolean computeChildren() {
    final var newChildren = new ArrayList<TreeNode>();
    final var subcircs = new ArrayList<Component>();
    for (final var comp : circuitState.getCircuit().getNonWires()) {
      if (comp.getFactory() instanceof SubcircuitFactory) {
        subcircs.add(comp);
      } else {
        final var toAdd = model.mapComponentToNode(comp);
        if (toAdd != null) {
          newChildren.add(toAdd);
        }
      }
    }
    newChildren.sort(new CompareByName());
    subcircs.sort(this);
    for (final var comp : subcircs) {
      final var factory = (SubcircuitFactory) comp.getFactory();
      final var state = factory.getSubstate(circuitState, comp);
      SimulationTreeCircuitNode toAdd = null;
      for (final var treeNode : children) {
        if (treeNode instanceof SimulationTreeCircuitNode node) {
          if (node.circuitState == state) {
            toAdd = node;
            break;
          }
        }
      }
      if (toAdd == null) {
        toAdd = new SimulationTreeCircuitNode(model, this, state, comp);
      }
      newChildren.add(toAdd);
    }

    if (!children.equals(newChildren)) {
      children = newChildren;
      return true;
    } else {
      return false;
    }
  }

  public CircuitState getCircuitState() {
    return circuitState;
  }

  @Override
  public ComponentFactory getComponentFactory() {
    return circuitState.getCircuit().getSubcircuitFactory();
  }

  @Override
  public boolean isCurrentView(SimulationTreeModel model) {
    return model.getCurrentView() == circuitState;
  }

  @Override
  public String toString() {
    if (subcircComp != null) {
      final var label = subcircComp.getAttributeSet().getValue(StdAttr.LABEL);
      if (label != null && !label.isEmpty()) {
        return label;
      }
    }
    var ret = circuitState.getCircuit().getName();
    if (subcircComp != null) {
      ret += subcircComp.getLocation();
    }
    return ret;
  }

  private static class CompareByName implements Comparator<Object> {
    @Override
    public int compare(Object a, Object b) {
      return a.toString().compareToIgnoreCase(b.toString());
    }
  }
}
