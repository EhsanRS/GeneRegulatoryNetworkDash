import cytoscape from 'cytoscape';
import type { Core, NodeSingular, EdgeSingular } from 'cytoscape';
import cola from 'cytoscape-cola';
import dagre from 'cytoscape-dagre';
import grnData from '../synth_grn.json';

// Register layout extensions
cytoscape.use(cola);
cytoscape.use(dagre);

// Types
interface GRNEdge {
  source: string;
  target: string;
  weight: number;
  type: string;
}

interface GRNData {
  gene_names: string[];
  edges: GRNEdge[];
  ligand_receptor_pairs: Array<{ pair_id: number; ligand: number; receptor: number }>;
}

// Color schemes
const nodeColors: Record<string, string> = {
  'TF_PROG': '#58a6ff',
  'TF_LIN': '#a371f7',
  'LIG': '#3fb950',
  'REC': '#39c5cf',
  'TARG': '#f85149',
  'HK': '#6e7681',
  'OTHER': '#d29922',
};

const edgeColors: Record<string, string> = {
  'prog_self': '#58a6ff',
  'prog_mutual': '#58a6ff',
  'prog_to_lineage': '#7c72ff',
  'lineage_self': '#a371f7',
  'lineage_partner': '#a371f7',
  'lineage_inhibit': '#f85149',
  'lineage_repress_prog': '#f85149',
  'lineage_to_target': '#db61a2',
  'lineage_to_ligand': '#3fb950',
  'prog_to_receptor': '#39c5cf',
  'random': '#484f58',
};

const edgeLabels: Record<string, string> = {
  'prog_self': 'Progenitor Self-activation',
  'prog_mutual': 'Progenitor Mutual Activation',
  'prog_to_lineage': 'Progenitor → Lineage',
  'lineage_self': 'Lineage Self-activation',
  'lineage_partner': 'Lineage Partner',
  'lineage_inhibit': 'Lineage Inhibition',
  'lineage_repress_prog': 'Lineage → Progenitor Repression',
  'lineage_to_target': 'Lineage → Target',
  'lineage_to_ligand': 'Lineage → Ligand',
  'prog_to_receptor': 'Progenitor → Receptor',
  'random': 'Random',
};

const geneTypeLabels: Record<string, string> = {
  'TF_PROG': 'Progenitor TFs',
  'TF_LIN': 'Lineage TFs',
  'LIG': 'Ligands',
  'REC': 'Receptors',
  'TARG': 'Targets',
  'HK': 'Housekeeping',
  'OTHER': 'Other',
};

const geneTypeOrder = ['TF_PROG', 'TF_LIN', 'LIG', 'REC', 'TARG', 'HK', 'OTHER'];

// Safe getters
const getNodeColor = (type: string): string => nodeColors[type] ?? '#8b949e';
const getEdgeColor = (type: string): string => edgeColors[type] ?? '#8b949e';
const getGeneTypeLabel = (type: string): string => geneTypeLabels[type] ?? type;

function getGeneType(name: string): string {
  if (name.startsWith('TF_PROG')) return 'TF_PROG';
  if (name.startsWith('TF_LIN')) return 'TF_LIN';
  if (name.startsWith('LIG')) return 'LIG';
  if (name.startsWith('REC')) return 'REC';
  if (name.startsWith('TARG')) return 'TARG';
  if (name.startsWith('HK')) return 'HK';
  if (name.startsWith('OTHER')) return 'OTHER';
  return 'OTHER';
}

// State
let cy: Core;
let activeGeneTypes = new Set(geneTypeOrder);
let activeEdgeTypes = new Set(Object.keys(edgeLabels));
let minWeight = 0;
let showLabels = true;
let currentLayout = 'cola';

// Helper
function createElementWithText(tag: string, text: string, className?: string): HTMLElement {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}

function init() {
  const data = grnData as GRNData;

  // Build nodes
  const nodeSet = new Set<string>();
  data.edges.forEach(e => {
    nodeSet.add(e.source);
    nodeSet.add(e.target);
  });

  const nodes = Array.from(nodeSet).map(id => ({
    data: {
      id,
      type: getGeneType(id),
      label: id,
    },
  }));

  // Build edges
  const edges = data.edges.map((e, i) => ({
    data: {
      id: `e${i}`,
      source: e.source,
      target: e.target,
      weight: e.weight,
      edgeType: e.type,
    },
  }));

  // Initialize Cytoscape
  cy = cytoscape({
    container: document.getElementById('network-container'),
    elements: { nodes, edges },
    style: [
      {
        selector: 'node',
        style: {
          'background-color': (ele: NodeSingular) => getNodeColor(ele.data('type')),
          'label': 'data(label)',
          'color': '#e6edf3',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'font-size': '10px',
          'text-margin-y': 5,
          'width': (ele: NodeSingular) => {
            const type = ele.data('type');
            if (type === 'TF_PROG') return 30;
            if (type === 'TF_LIN') return 24;
            return 16;
          },
          'height': (ele: NodeSingular) => {
            const type = ele.data('type');
            if (type === 'TF_PROG') return 30;
            if (type === 'TF_LIN') return 24;
            return 16;
          },
          'border-width': 2,
          'border-color': '#0d1117',
        },
      },
      {
        selector: 'node:selected',
        style: {
          'border-width': 4,
          'border-color': '#e3b341',
        },
      },
      {
        selector: 'edge',
        style: {
          'width': (ele: EdgeSingular) => Math.max(1, Math.abs(ele.data('weight')) * 3),
          'line-color': (ele: EdgeSingular) => getEdgeColor(ele.data('edgeType')),
          'target-arrow-color': (ele: EdgeSingular) => getEdgeColor(ele.data('edgeType')),
          'target-arrow-shape': (ele: EdgeSingular) => ele.data('weight') < 0 ? 'tee' : 'triangle',
          'curve-style': 'bezier',
          'opacity': (ele: EdgeSingular) => Math.min(0.8, 0.3 + Math.abs(ele.data('weight')) * 0.5),
          'line-style': (ele: EdgeSingular) => ele.data('weight') < 0 ? 'dashed' : 'solid',
        },
      },
      {
        selector: 'edge:selected',
        style: {
          'width': 4,
          'opacity': 1,
        },
      },
      {
        selector: '.highlighted',
        style: {
          'opacity': 1,
          'z-index': 999,
        },
      },
      {
        selector: '.faded',
        style: {
          'opacity': 0.1,
        },
      },
      {
        selector: '.hidden',
        style: {
          'display': 'none',
        },
      },
    ],
    layout: {
      name: 'cola',
      nodeSpacing: 40,
      edgeLengthVal: 80,
      animate: false,
      randomize: true,
      maxSimulationTime: 4000,
    } as cytoscape.LayoutOptions,
    wheelSensitivity: 0.3,
    minZoom: 0.1,
    maxZoom: 5,
  });

  // Event handlers
  cy.on('tap', 'node', (evt) => {
    const node = evt.target;
    selectNode(node);
  });

  cy.on('tap', (evt) => {
    if (evt.target === cy) {
      clearSelection();
    }
  });

  cy.on('mouseover', 'node', (evt) => {
    showTooltip(evt);
  });

  cy.on('mouseout', 'node', () => {
    hideTooltip();
  });

  // Setup UI
  updateStats();
  setupFilters();
  setupControls();
  setupLegend();
  updateVisibility();
}

function updateVisibility() {
  // Update node visibility
  cy.nodes().forEach(node => {
    const type = node.data('type');
    if (activeGeneTypes.has(type)) {
      node.removeClass('hidden');
    } else {
      node.addClass('hidden');
    }
  });

  // Update edge visibility
  cy.edges().forEach(edge => {
    const edgeType = edge.data('edgeType');
    const weight = Math.abs(edge.data('weight'));
    const sourceVisible = !edge.source().hasClass('hidden');
    const targetVisible = !edge.target().hasClass('hidden');

    if (activeEdgeTypes.has(edgeType) && weight >= minWeight && sourceVisible && targetVisible) {
      edge.removeClass('hidden');
    } else {
      edge.addClass('hidden');
    }
  });

  updateStats();
}

function selectNode(node: NodeSingular) {
  // Clear previous highlighting
  cy.elements().removeClass('highlighted faded');

  // Get connected elements
  const neighborhood = node.neighborhood().add(node);

  // Highlight neighborhood, fade others
  neighborhood.addClass('highlighted');
  cy.elements().not(neighborhood).addClass('faded');

  // Update details panel
  updateNodeDetails(node);
}

function clearSelection() {
  cy.elements().removeClass('highlighted faded');
  const details = document.getElementById('node-details')!;
  details.replaceChildren();
  const placeholder = createElementWithText('p', 'Click a node to see details', 'placeholder');
  details.appendChild(placeholder);
}

function showTooltip(evt: cytoscape.EventObject) {
  const node = evt.target;
  const tooltip = document.getElementById('tooltip')!;

  const inDegree = node.incomers('edge').filter((e: EdgeSingular) => !e.hasClass('hidden')).length;
  const outDegree = node.outgoers('edge').filter((e: EdgeSingular) => !e.hasClass('hidden')).length;

  tooltip.replaceChildren();
  const title = createElementWithText('div', node.data('id'), 'tooltip-title');
  const type = createElementWithText('div', getGeneTypeLabel(node.data('type')), 'tooltip-type');
  const connections = createElementWithText('div', `In: ${inDegree} | Out: ${outDegree}`);
  connections.style.marginTop = '0.5rem';
  connections.style.fontSize = '0.75rem';
  connections.style.color = 'var(--text-secondary)';
  tooltip.appendChild(title);
  tooltip.appendChild(type);
  tooltip.appendChild(connections);

  const pos = evt.renderedPosition;
  const container = document.getElementById('network-container')!;
  const rect = container.getBoundingClientRect();

  tooltip.style.left = `${rect.left + pos.x + 15}px`;
  tooltip.style.top = `${rect.top + pos.y - 10}px`;
  tooltip.classList.add('visible');
}

function hideTooltip() {
  const tooltip = document.getElementById('tooltip')!;
  tooltip.classList.remove('visible');
}

function updateNodeDetails(node: NodeSingular) {
  const details = document.getElementById('node-details')!;
  details.replaceChildren();

  const inEdges = node.incomers('edge').filter((e: EdgeSingular) => !e.hasClass('hidden'));
  const outEdges = node.outgoers('edge').filter((e: EdgeSingular) => !e.hasClass('hidden'));

  // Gene name
  const geneItem = document.createElement('div');
  geneItem.className = 'node-detail-item';
  const geneLabel = createElementWithText('div', 'Gene', 'node-detail-label');
  const geneValue = createElementWithText('div', node.data('id'), 'node-detail-value');
  geneValue.style.color = getNodeColor(node.data('type'));
  geneItem.appendChild(geneLabel);
  geneItem.appendChild(geneValue);
  details.appendChild(geneItem);

  // Type
  const typeItem = document.createElement('div');
  typeItem.className = 'node-detail-item';
  typeItem.appendChild(createElementWithText('div', 'Type', 'node-detail-label'));
  typeItem.appendChild(createElementWithText('div', getGeneTypeLabel(node.data('type')), 'node-detail-value'));
  details.appendChild(typeItem);

  // Incoming
  const inItem = document.createElement('div');
  inItem.className = 'node-detail-item';
  inItem.appendChild(createElementWithText('div', 'Incoming Edges', 'node-detail-label'));
  inItem.appendChild(createElementWithText('div', String(inEdges.length), 'node-detail-value'));
  details.appendChild(inItem);

  // Outgoing
  const outItem = document.createElement('div');
  outItem.className = 'node-detail-item';
  outItem.appendChild(createElementWithText('div', 'Outgoing Edges', 'node-detail-label'));
  outItem.appendChild(createElementWithText('div', String(outEdges.length), 'node-detail-value'));
  details.appendChild(outItem);

  // Regulators
  const regItem = document.createElement('div');
  regItem.className = 'node-detail-item';
  regItem.appendChild(createElementWithText('div', 'Regulators (top 10)', 'node-detail-label'));
  const regValue = document.createElement('div');
  regValue.className = 'node-detail-value';
  regValue.style.fontSize = '0.75rem';
  regValue.style.lineHeight = '1.6';

  if (inEdges.length === 0) {
    regValue.textContent = 'None';
  } else {
    inEdges.slice(0, 10).forEach((edge: EdgeSingular, i: number) => {
      const sourceId = edge.source().data('id');
      const weight = edge.data('weight');
      const span = document.createElement('span');
      span.textContent = sourceId;
      span.style.color = weight > 0 ? 'var(--accent-green)' : 'var(--accent-red)';
      regValue.appendChild(span);
      if (i < Math.min(inEdges.length, 10) - 1) {
        regValue.appendChild(document.createTextNode(', '));
      }
    });
    if (inEdges.length > 10) {
      const more = document.createElement('span');
      more.textContent = ` ... +${inEdges.length - 10} more`;
      more.style.color = 'var(--text-secondary)';
      regValue.appendChild(more);
    }
  }
  regItem.appendChild(regValue);
  details.appendChild(regItem);

  // Targets
  const targetItem = document.createElement('div');
  targetItem.className = 'node-detail-item';
  targetItem.appendChild(createElementWithText('div', 'Targets (top 10)', 'node-detail-label'));
  const targetValue = document.createElement('div');
  targetValue.className = 'node-detail-value';
  targetValue.style.fontSize = '0.75rem';
  targetValue.style.lineHeight = '1.6';

  if (outEdges.length === 0) {
    targetValue.textContent = 'None';
  } else {
    outEdges.slice(0, 10).forEach((edge: EdgeSingular, i: number) => {
      const targetId = edge.target().data('id');
      const weight = edge.data('weight');
      const span = document.createElement('span');
      span.textContent = targetId;
      span.style.color = weight > 0 ? 'var(--accent-green)' : 'var(--accent-red)';
      targetValue.appendChild(span);
      if (i < Math.min(outEdges.length, 10) - 1) {
        targetValue.appendChild(document.createTextNode(', '));
      }
    });
    if (outEdges.length > 10) {
      const more = document.createElement('span');
      more.textContent = ` ... +${outEdges.length - 10} more`;
      more.style.color = 'var(--text-secondary)';
      targetValue.appendChild(more);
    }
  }
  targetItem.appendChild(targetValue);
  details.appendChild(targetItem);
}

function updateStats() {
  const visibleNodes = cy.nodes().filter((n: NodeSingular) => !n.hasClass('hidden'));
  const visibleEdges = cy.edges().filter((e: EdgeSingular) => !e.hasClass('hidden'));

  const activating = visibleEdges.filter((e: EdgeSingular) => e.data('weight') > 0).length;
  const inhibiting = visibleEdges.filter((e: EdgeSingular) => e.data('weight') < 0).length;

  document.getElementById('stat-genes')!.textContent = String(visibleNodes.length);
  document.getElementById('stat-edges')!.textContent = String(visibleEdges.length);
  document.getElementById('stat-activating')!.textContent = String(activating);
  document.getElementById('stat-inhibiting')!.textContent = String(inhibiting);
}

function setupFilters() {
  // Gene type filters
  const geneFilters = document.getElementById('gene-filters')!;
  const geneCounts: Record<string, number> = {};
  cy.nodes().forEach((n: NodeSingular) => {
    const type = n.data('type');
    geneCounts[type] = (geneCounts[type] ?? 0) + 1;
  });

  geneTypeOrder.forEach(type => {
    const count = geneCounts[type] ?? 0;
    const item = document.createElement('label');
    item.className = 'filter-item';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;

    const colorSpan = document.createElement('span');
    colorSpan.className = 'filter-color';
    colorSpan.style.background = getNodeColor(type);

    const labelSpan = createElementWithText('span', getGeneTypeLabel(type), 'filter-label');
    const countSpan = createElementWithText('span', String(count), 'filter-count');

    item.appendChild(checkbox);
    item.appendChild(colorSpan);
    item.appendChild(labelSpan);
    item.appendChild(countSpan);
    geneFilters.appendChild(item);

    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        activeGeneTypes.add(type);
      } else {
        activeGeneTypes.delete(type);
      }
      updateVisibility();
    });
  });

  // Edge type filters
  const edgeFilters = document.getElementById('edge-filters')!;
  const edgeCounts: Record<string, number> = {};
  cy.edges().forEach((e: EdgeSingular) => {
    const type = e.data('edgeType');
    edgeCounts[type] = (edgeCounts[type] ?? 0) + 1;
  });

  Object.entries(edgeLabels).forEach(([type, label]) => {
    const count = edgeCounts[type] ?? 0;
    const item = document.createElement('label');
    item.className = 'filter-item';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;

    const colorSpan = document.createElement('span');
    colorSpan.className = 'filter-color';
    colorSpan.style.background = getEdgeColor(type);

    const labelSpan = createElementWithText('span', label, 'filter-label');
    const countSpan = createElementWithText('span', String(count), 'filter-count');

    item.appendChild(checkbox);
    item.appendChild(colorSpan);
    item.appendChild(labelSpan);
    item.appendChild(countSpan);
    edgeFilters.appendChild(item);

    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        activeEdgeTypes.add(type);
      } else {
        activeEdgeTypes.delete(type);
      }
      updateVisibility();
    });
  });
}

function setupControls() {
  // Show labels
  document.getElementById('show-labels')!.addEventListener('change', (e) => {
    showLabels = (e.target as HTMLInputElement).checked;
    cy.style().selector('node').style('label', showLabels ? 'data(label)' : '').update();
  });

  // Min weight slider
  const minWeightSlider = document.getElementById('min-weight') as HTMLInputElement;
  const minWeightVal = document.getElementById('min-weight-val')!;
  minWeightSlider.addEventListener('input', () => {
    minWeight = parseFloat(minWeightSlider.value);
    minWeightVal.textContent = minWeight.toFixed(2);
    updateVisibility();
  });

  // Layout buttons
  document.getElementById('layout-cola')!.addEventListener('click', () => applyLayout('cola'));
  document.getElementById('layout-dagre')!.addEventListener('click', () => applyLayout('dagre'));
  document.getElementById('layout-concentric')!.addEventListener('click', () => applyLayout('concentric'));

  // Reset zoom
  document.getElementById('reset-zoom')!.addEventListener('click', () => {
    cy.fit(undefined, 50);
  });

  // Set initial active button
  document.getElementById('layout-cola')!.classList.add('active');
}

function applyLayout(layout: string) {
  currentLayout = layout;

  // Update active button
  document.querySelectorAll('.button-group button').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`layout-${layout}`)?.classList.add('active');

  let layoutConfig: cytoscape.LayoutOptions;

  switch (layout) {
    case 'cola':
      layoutConfig = {
        name: 'cola',
        nodeSpacing: 40,
        edgeLengthVal: 80,
        animate: true,
        randomize: false,
        maxSimulationTime: 3000,
      } as cytoscape.LayoutOptions;
      break;

    case 'dagre':
      layoutConfig = {
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 50,
        rankSep: 100,
        animate: true,
      } as cytoscape.LayoutOptions;
      break;

    case 'concentric':
      layoutConfig = {
        name: 'concentric',
        concentric: (node: NodeSingular) => {
          const type = node.data('type');
          const order = geneTypeOrder.indexOf(type);
          return geneTypeOrder.length - order;
        },
        levelWidth: () => 2,
        minNodeSpacing: 30,
        animate: true,
      } as cytoscape.LayoutOptions;
      break;

    default:
      return;
  }

  cy.layout(layoutConfig).run();
}

function setupLegend() {
  const legend = document.getElementById('legend')!;

  // Node types
  const nodeSection = document.createElement('div');
  nodeSection.className = 'legend-section';
  const nodeHeader = createElementWithText('h3', 'Gene Types');
  nodeSection.appendChild(nodeHeader);

  geneTypeOrder.forEach(type => {
    const item = document.createElement('div');
    item.className = 'legend-item';

    const nodeCircle = document.createElement('span');
    nodeCircle.className = 'legend-node';
    nodeCircle.style.background = getNodeColor(type);

    const label = createElementWithText('span', getGeneTypeLabel(type));

    item.appendChild(nodeCircle);
    item.appendChild(label);
    nodeSection.appendChild(item);
  });
  legend.appendChild(nodeSection);

  // Edge types
  const edgeSection = document.createElement('div');
  edgeSection.className = 'legend-section';
  const edgeHeader = createElementWithText('h3', 'Edge Types');
  edgeSection.appendChild(edgeHeader);

  const activatingItem = document.createElement('div');
  activatingItem.className = 'legend-item';
  const activatingLine = document.createElement('span');
  activatingLine.className = 'legend-edge';
  activatingLine.style.background = 'var(--accent-green)';
  activatingItem.appendChild(activatingLine);
  activatingItem.appendChild(createElementWithText('span', 'Activation'));
  edgeSection.appendChild(activatingItem);

  const inhibitingItem = document.createElement('div');
  inhibitingItem.className = 'legend-item';
  const inhibitingLine = document.createElement('span');
  inhibitingLine.className = 'legend-edge dashed';
  inhibitingLine.style.color = 'var(--accent-red)';
  inhibitingItem.appendChild(inhibitingLine);
  inhibitingItem.appendChild(createElementWithText('span', 'Inhibition'));
  edgeSection.appendChild(inhibitingItem);

  legend.appendChild(edgeSection);
}

// Start
document.addEventListener('DOMContentLoaded', init);
