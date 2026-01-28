/**
 * Parameter Inference Page
 *
 * Allows users to define a target cell state and uses CMA-ES optimization
 * to find simulation parameters that produce that state.
 */

import grnData from '../synth_grn.json';
import { InferenceEngine, type GRNData, type GeneGroups, type StabilityResult } from './inference/inference-engine';
import {
  decode,
  encode,
  getDimensions,
  getParameterNames,
  type SimulationParams,
  type EncodingConfig,
  exportParamsJSON,
  encodeParamsURL,
  GLOBAL_PARAM_BOUNDS,
  GLOBAL_PARAM_KEYS,
} from './inference/encoding';
import { computeFitness, type TargetState } from './inference/objective';
import type {
  WorkerStartMessage,
  WorkerOutMessage,
  WorkerProgressMessage,
  WorkerDoneMessage,
  TimePointTarget,
} from './inference/worker';

// Initialize inference engine
const data = grnData as GRNData;
const engine = new InferenceEngine(data);
const geneNames = engine.getGeneNames();
const geneGroups = engine.getGeneGroups();
const nGenes = engine.getNumGenes();
const knockoutCandidates = engine.getKnockoutCandidates();

// State
let worker: Worker | null = null;
let isRunning = false;
let targetExpression = new Float32Array(nGenes);
let targetWeights = new Float32Array(nGenes).fill(1);
let bestParams: SimulationParams | null = null;

// Chart state
let fitnessHistory: { gen: number; fitness: number; meanFitness: number }[] = [];
let bestSimulatedExpression: Float32Array | null = null;

// Sensitivity analysis state
let sensitivityResults: { paramName: string; sensitivity: number }[] = [];
let sensitivityCanvas: HTMLCanvasElement;
let sensitivityCtx: CanvasRenderingContext2D;

// Stability analysis state
let stabilityResult: StabilityResult | null = null;

// Canvas contexts
let fitnessCanvas: HTMLCanvasElement;
let fitnessCtx: CanvasRenderingContext2D;
let comparisonCanvas: HTMLCanvasElement;
let comparisonCtx: CanvasRenderingContext2D;
let scatterCanvas: HTMLCanvasElement;
let scatterCtx: CanvasRenderingContext2D;
let errorCanvas: HTMLCanvasElement;
let errorCtx: CanvasRenderingContext2D;

// Weight preset modes
let weightPreset: 'uniform' | 'tfs-only' | 'select' | 'custom' = 'select';

// Selected genes for 'select' mode
const selectedGenes = new Set<number>();

// Active state preset for highlighting
let activeStatePreset: string | null = 'progenitor';

// Time-course mode state
let timeCourseMode = false;
let timeCourseTargets: TimePointTarget[] = [];
let nextTimepointId = 1;
let editingTimepointId: number | null = null;

// Cell state presets - predefined expression patterns for common cell states
const CELL_STATE_PRESETS: Record<string, {
  description: string;
  progTF: number;
  linTF: { [key: number]: number };  // Lineage index (0-5) -> expression level
  defaultLinTF: number;
  ligand: number;
  receptor: number;
}> = {
  progenitor: {
    description: 'Undifferentiated progenitor state',
    progTF: 2.0,
    linTF: {},
    defaultLinTF: 0.2,
    ligand: 0.3,
    receptor: 0.3,
  },
  l1: {
    description: 'Lineage 1 committed',
    progTF: 0.3,
    linTF: { 0: 4.0 },  // L1 TFs high
    defaultLinTF: 0.2,
    ligand: 0.5,
    receptor: 0.5,
  },
  l2: {
    description: 'Lineage 2 committed',
    progTF: 0.3,
    linTF: { 1: 4.0 },  // L2 TFs high
    defaultLinTF: 0.2,
    ligand: 0.5,
    receptor: 0.5,
  },
  l3: {
    description: 'Lineage 3 committed',
    progTF: 0.3,
    linTF: { 2: 4.0 },  // L3 TFs high
    defaultLinTF: 0.2,
    ligand: 0.5,
    receptor: 0.5,
  },
  l4: {
    description: 'Lineage 4 committed',
    progTF: 0.3,
    linTF: { 3: 4.0 },  // L4 TFs high
    defaultLinTF: 0.2,
    ligand: 0.5,
    receptor: 0.5,
  },
  l5: {
    description: 'Lineage 5 committed',
    progTF: 0.3,
    linTF: { 4: 4.0 },  // L5 TFs high
    defaultLinTF: 0.2,
    ligand: 0.5,
    receptor: 0.5,
  },
  l6: {
    description: 'Lineage 6 committed',
    progTF: 0.3,
    linTF: { 5: 4.0 },  // L6 TFs high
    defaultLinTF: 0.2,
    ligand: 0.5,
    receptor: 0.5,
  },
  multipotent: {
    description: 'Poised for multiple fates',
    progTF: 1.0,
    linTF: {},
    defaultLinTF: 1.0,  // Moderate expression of all lineage TFs
    ligand: 0.5,
    receptor: 0.5,
  },
};

// Initialize default target expression (progenitor state)
function initializeDefaultTarget(): void {
  // Start with moderate progenitor expression
  geneGroups.tf_prog.forEach(i => { targetExpression[i] = 1.5; });
  // Low lineage TF expression
  geneGroups.tf_lin.forEach(i => { targetExpression[i] = 0.2; });
  // Moderate ligand/receptor
  geneGroups.ligand.forEach(i => { targetExpression[i] = 0.3; });
  geneGroups.receptor.forEach(i => { targetExpression[i] = 0.3; });
  // Low targets
  geneGroups.target.forEach(i => { targetExpression[i] = 0.1; });
  // Housekeeping stable
  geneGroups.housekeeping.forEach(i => { targetExpression[i] = 1.0; });
  geneGroups.other.forEach(i => { targetExpression[i] = 0.1; });

  // Pre-select all TFs (prog and lin) for weighting
  geneGroups.tf_prog.forEach(i => { selectedGenes.add(i); });
  geneGroups.tf_lin.forEach(i => { selectedGenes.add(i); });
}

// Apply a cell state preset to target expression
function applyStatePreset(presetName: string): void {
  const preset = CELL_STATE_PRESETS[presetName];
  if (!preset) return;

  // Set progenitor TF expression
  geneGroups.tf_prog.forEach(i => {
    targetExpression[i] = preset.progTF;
  });

  // Set lineage TF expression
  // Each lineage has 2 TFs (tfsPerLineage = 2)
  const tfsPerLineage = 2;
  const nLineages = 6;

  for (let lin = 0; lin < nLineages; lin++) {
    const startIdx = lin * tfsPerLineage;
    const endIdx = Math.min((lin + 1) * tfsPerLineage, geneGroups.tf_lin.length);

    // Check if this lineage has a specific expression level
    const level = preset.linTF[lin] ?? preset.defaultLinTF;

    for (let i = startIdx; i < endIdx; i++) {
      const geneIdx = geneGroups.tf_lin[i];
      if (geneIdx !== undefined) {
        targetExpression[geneIdx] = level;
      }
    }
  }

  // Set ligand and receptor expression
  geneGroups.ligand.forEach(i => { targetExpression[i] = preset.ligand; });
  geneGroups.receptor.forEach(i => { targetExpression[i] = preset.receptor; });

  // Update active preset indicator
  activeStatePreset = presetName;

  // Update slider values in the UI
  updateTargetSliderValues();

  // Update charts
  updateCharts();
}

// Update all target slider values from targetExpression array
function updateTargetSliderValues(): void {
  const allIndices = [
    ...geneGroups.tf_prog,
    ...geneGroups.tf_lin,
    ...geneGroups.ligand,
    ...geneGroups.receptor,
  ];

  for (const idx of allIndices) {
    const slider = document.getElementById(`target-${idx}`) as HTMLInputElement;
    const valSpan = document.getElementById(`val-target-${idx}`);

    if (slider && valSpan) {
      slider.value = targetExpression[idx].toFixed(1);
      valSpan.textContent = targetExpression[idx].toFixed(1);
    }
  }
}

// Update state preset button styling
function updateStatePresetButtons(): void {
  const buttons = document.querySelectorAll('.state-preset-btn');
  buttons.forEach(btn => {
    const preset = btn.getAttribute('data-preset');
    btn.classList.toggle('active', preset === activeStatePreset);
  });
}

// Initialize state preset buttons
function initializeStatePresets(): void {
  const buttons = document.querySelectorAll('.state-preset-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const presetName = btn.getAttribute('data-preset');
      if (presetName) {
        applyStatePreset(presetName);
        updateStatePresetButtons();
      }
    });
  });

  // Set initial active state
  updateStatePresetButtons();
}

// Time-course mode management

// Switch between Final State and Time Course modes
function switchTargetMode(mode: 'final' | 'timecourse'): void {
  timeCourseMode = mode === 'timecourse';

  // Update tab buttons
  const tabs = document.querySelectorAll('.target-mode-tab');
  tabs.forEach(tab => {
    const tabMode = tab.getAttribute('data-mode');
    tab.classList.toggle('active', tabMode === mode);
  });

  // Show/hide panels
  const finalPanel = document.getElementById('final-state-panel');
  const timecoursePanel = document.getElementById('timecourse-panel');

  if (finalPanel) finalPanel.classList.toggle('hidden', mode !== 'final');
  if (timecoursePanel) timecoursePanel.classList.toggle('hidden', mode !== 'timecourse');
}

// Create preset select options
function createPresetSelectOptions(selectEl: HTMLSelectElement): void {
  const options = [
    { value: '', label: 'Custom' },
    { value: 'progenitor', label: 'Progenitor' },
    { value: 'multipotent', label: 'Multipotent' },
    { value: 'l1', label: 'L1 Committed' },
    { value: 'l2', label: 'L2 Committed' },
    { value: 'l3', label: 'L3 Committed' },
    { value: 'l4', label: 'L4 Committed' },
    { value: 'l5', label: 'L5 Committed' },
    { value: 'l6', label: 'L6 Committed' },
  ];

  for (const opt of options) {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    selectEl.appendChild(option);
  }
}

// Create a timepoint entry in the list
function createTimepointEntry(timepoint: TimePointTarget & { id: number }): HTMLElement {
  const entry = document.createElement('div');
  entry.className = 'timepoint-entry';
  entry.dataset.timepointId = String(timepoint.id);

  // Header row with time and preset selector
  const header = document.createElement('div');
  header.className = 'timepoint-header';

  const timeLabel = document.createElement('span');
  timeLabel.className = 'timepoint-time';
  timeLabel.textContent = `t = ${timepoint.time}h`;

  const presetSelect = document.createElement('select');
  presetSelect.className = 'timepoint-preset-select';
  createPresetSelectOptions(presetSelect);

  presetSelect.addEventListener('change', () => {
    const presetName = presetSelect.value;
    if (presetName && CELL_STATE_PRESETS[presetName]) {
      applyPresetToTimepoint(timepoint.id, presetName);
    }
  });

  const editBtn = document.createElement('button');
  editBtn.className = 'timepoint-edit-btn';
  editBtn.textContent = 'Edit';
  editBtn.title = 'Edit expression levels';
  editBtn.addEventListener('click', () => openTimepointModal(timepoint.id));

  const removeBtn = document.createElement('button');
  removeBtn.className = 'timepoint-remove-btn';
  removeBtn.textContent = '×';
  removeBtn.title = 'Remove timepoint';
  removeBtn.addEventListener('click', () => removeTimepoint(timepoint.id));

  header.appendChild(timeLabel);
  header.appendChild(presetSelect);
  header.appendChild(editBtn);
  header.appendChild(removeBtn);

  // Weight slider
  const weightRow = document.createElement('div');
  weightRow.className = 'timepoint-weight-row';

  const weightLabel = document.createElement('label');
  weightLabel.textContent = 'Weight:';

  const weightSlider = document.createElement('input');
  weightSlider.type = 'range';
  weightSlider.min = '0.1';
  weightSlider.max = '2';
  weightSlider.step = '0.1';
  weightSlider.value = String(timepoint.weight);
  weightSlider.className = 'timepoint-weight-slider';

  const weightVal = document.createElement('span');
  weightVal.className = 'timepoint-weight-val';
  weightVal.textContent = timepoint.weight.toFixed(1);

  weightSlider.addEventListener('input', () => {
    const weight = parseFloat(weightSlider.value);
    weightVal.textContent = weight.toFixed(1);
    updateTimepointWeight(timepoint.id, weight);
  });

  weightRow.appendChild(weightLabel);
  weightRow.appendChild(weightSlider);
  weightRow.appendChild(weightVal);

  entry.appendChild(header);
  entry.appendChild(weightRow);

  return entry;
}

// Add a new timepoint
function addTimepoint(time?: number, presetName?: string): void {
  // Default time: after last timepoint or 0 if first
  const defaultTime = timeCourseTargets.length > 0
    ? Math.max(...timeCourseTargets.map(t => t.time)) + 4
    : 0;

  const newTime = time ?? defaultTime;
  const id = nextTimepointId++;

  // Create expression array based on preset or current target
  let expression: number[];
  if (presetName && CELL_STATE_PRESETS[presetName]) {
    expression = generateExpressionFromPreset(presetName);
  } else {
    expression = Array.from(targetExpression);
  }

  const timepoint: TimePointTarget & { id: number } = {
    id,
    time: newTime,
    expression,
    weight: 1.0,
  };

  timeCourseTargets.push(timepoint);

  // Add to UI
  const list = document.getElementById('timepoint-list');
  if (list) {
    const entry = createTimepointEntry(timepoint);
    list.appendChild(entry);
  }

  // Sort timepoints by time
  sortTimepoints();
}

// Generate expression array from a preset
function generateExpressionFromPreset(presetName: string): number[] {
  const preset = CELL_STATE_PRESETS[presetName];
  if (!preset) return Array.from(targetExpression);

  const expr = new Float32Array(nGenes);

  // Set progenitor TF expression
  geneGroups.tf_prog.forEach(i => {
    expr[i] = preset.progTF;
  });

  // Set lineage TF expression
  const tfsPerLineage = 2;
  const nLineages = 6;

  for (let lin = 0; lin < nLineages; lin++) {
    const startIdx = lin * tfsPerLineage;
    const endIdx = Math.min((lin + 1) * tfsPerLineage, geneGroups.tf_lin.length);
    const level = preset.linTF[lin] ?? preset.defaultLinTF;

    for (let i = startIdx; i < endIdx; i++) {
      const geneIdx = geneGroups.tf_lin[i];
      if (geneIdx !== undefined) {
        expr[geneIdx] = level;
      }
    }
  }

  // Set ligand and receptor expression
  geneGroups.ligand.forEach(i => { expr[i] = preset.ligand; });
  geneGroups.receptor.forEach(i => { expr[i] = preset.receptor; });

  // Set other genes to low values
  geneGroups.target.forEach(i => { expr[i] = 0.1; });
  geneGroups.housekeeping.forEach(i => { expr[i] = 1.0; });
  geneGroups.other.forEach(i => { expr[i] = 0.1; });

  return Array.from(expr);
}

// Apply a preset to a specific timepoint
function applyPresetToTimepoint(timepointId: number, presetName: string): void {
  const timepoint = timeCourseTargets.find(t => (t as TimePointTarget & { id: number }).id === timepointId);
  if (timepoint) {
    timepoint.expression = generateExpressionFromPreset(presetName);
  }
}

// Update timepoint weight
function updateTimepointWeight(timepointId: number, weight: number): void {
  const timepoint = timeCourseTargets.find(t => (t as TimePointTarget & { id: number }).id === timepointId);
  if (timepoint) {
    timepoint.weight = weight;
  }
}

// Remove a timepoint
function removeTimepoint(timepointId: number): void {
  const idx = timeCourseTargets.findIndex(t => (t as TimePointTarget & { id: number }).id === timepointId);
  if (idx >= 0) {
    timeCourseTargets.splice(idx, 1);
  }

  // Remove from UI
  const entry = document.querySelector(`.timepoint-entry[data-timepoint-id="${timepointId}"]`);
  entry?.remove();
}

// Sort timepoints by time and re-render
function sortTimepoints(): void {
  timeCourseTargets.sort((a, b) => a.time - b.time);

  // Re-render the list
  const list = document.getElementById('timepoint-list');
  if (!list) return;

  // Clear existing entries
  while (list.firstChild) {
    list.removeChild(list.firstChild);
  }

  // Re-add sorted entries
  for (const tp of timeCourseTargets) {
    const entry = createTimepointEntry(tp as TimePointTarget & { id: number });
    list.appendChild(entry);
  }
}

// Initialize time-course UI
function initializeTimeCourseUI(): void {
  // Set up mode tab buttons
  const tabs = document.querySelectorAll('.target-mode-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const mode = tab.getAttribute('data-mode') as 'final' | 'timecourse';
      if (mode) switchTargetMode(mode);
    });
  });

  // Set up add timepoint button
  const addBtn = document.getElementById('btn-add-timepoint');
  addBtn?.addEventListener('click', () => {
    // Show time picker dialog
    showAddTimepointDialog();
  });

  // Initialize with final state mode
  switchTargetMode('final');
}

// Show dialog to add a timepoint
function showAddTimepointDialog(): void {
  // Create a simple prompt for now (can be enhanced with a proper modal)
  const defaultTime = timeCourseTargets.length > 0
    ? Math.max(...timeCourseTargets.map(t => t.time)) + 4
    : 0;

  const timeStr = prompt(`Enter time (hours):`, String(defaultTime));
  if (timeStr === null) return;

  const time = parseFloat(timeStr);
  if (isNaN(time) || time < 0) {
    alert('Please enter a valid time (non-negative number)');
    return;
  }

  // Check for duplicate times
  if (timeCourseTargets.some(t => t.time === time)) {
    alert('A timepoint at this time already exists');
    return;
  }

  addTimepoint(time, 'progenitor');  // Default to progenitor preset
}

// Open timepoint edit modal
function openTimepointModal(timepointId: number): void {
  const timepoint = timeCourseTargets.find(t => (t as TimePointTarget & { id: number }).id === timepointId);
  if (!timepoint) return;

  editingTimepointId = timepointId;

  // Update modal title
  const titleEl = document.getElementById('timepoint-modal-title');
  if (titleEl) titleEl.textContent = `t = ${timepoint.time}h`;

  // Regenerate sliders for currently selected genes only
  regenerateTimepointModalSliders();

  // Populate modal sliders with timepoint's expression values
  populateTimepointModalSliders(timepoint.expression);

  // Show modal
  const modal = document.getElementById('timepoint-modal');
  modal?.classList.remove('hidden');
}

// Close timepoint modal
function closeTimepointModal(): void {
  const modal = document.getElementById('timepoint-modal');
  modal?.classList.add('hidden');
  editingTimepointId = null;
}

// Save timepoint modal values
function saveTimepointModal(): void {
  if (editingTimepointId === null) return;

  const timepoint = timeCourseTargets.find(t => (t as TimePointTarget & { id: number }).id === editingTimepointId);
  if (!timepoint) return;

  // Read expression values from modal sliders
  timepoint.expression = readTimepointModalSliders();

  // Update the preset dropdown to "Custom" since values may have changed
  const entry = document.querySelector(`.timepoint-entry[data-timepoint-id="${editingTimepointId}"]`);
  const presetSelect = entry?.querySelector('.timepoint-preset-select') as HTMLSelectElement;
  if (presetSelect) presetSelect.value = '';

  closeTimepointModal();
}

// Populate modal sliders with expression values
function populateTimepointModalSliders(expression: number[]): void {
  // Progenitor TFs
  geneGroups.tf_prog.forEach((geneIdx, i) => {
    const slider = document.getElementById(`tp-target-${geneIdx}`) as HTMLInputElement;
    const valSpan = document.getElementById(`tp-val-target-${geneIdx}`);
    if (slider && valSpan) {
      const val = expression[geneIdx] ?? 1.0;
      slider.value = String(val);
      valSpan.textContent = val.toFixed(1);
    }
  });

  // Lineage TFs
  geneGroups.tf_lin.forEach((geneIdx, i) => {
    const slider = document.getElementById(`tp-target-${geneIdx}`) as HTMLInputElement;
    const valSpan = document.getElementById(`tp-val-target-${geneIdx}`);
    if (slider && valSpan) {
      const val = expression[geneIdx] ?? 0.5;
      slider.value = String(val);
      valSpan.textContent = val.toFixed(1);
    }
  });

  // Ligands
  geneGroups.ligand.forEach((geneIdx, i) => {
    const slider = document.getElementById(`tp-target-${geneIdx}`) as HTMLInputElement;
    const valSpan = document.getElementById(`tp-val-target-${geneIdx}`);
    if (slider && valSpan) {
      const val = expression[geneIdx] ?? 0.5;
      slider.value = String(val);
      valSpan.textContent = val.toFixed(1);
    }
  });

  // Receptors
  geneGroups.receptor.forEach((geneIdx, i) => {
    const slider = document.getElementById(`tp-target-${geneIdx}`) as HTMLInputElement;
    const valSpan = document.getElementById(`tp-val-target-${geneIdx}`);
    if (slider && valSpan) {
      const val = expression[geneIdx] ?? 0.5;
      slider.value = String(val);
      valSpan.textContent = val.toFixed(1);
    }
  });
}

// Read expression values from modal sliders
function readTimepointModalSliders(): number[] {
  // Start with the current timepoint's expression (preserve non-selected genes)
  const timepoint = timeCourseTargets.find(t => (t as TimePointTarget & { id: number }).id === editingTimepointId);
  const expression = timepoint ? [...timepoint.expression] : new Array(nGenes).fill(0.1);

  // Read values only from selected genes (those with sliders)
  const allGenes = [
    ...geneGroups.tf_prog,
    ...geneGroups.tf_lin,
    ...geneGroups.ligand,
    ...geneGroups.receptor,
  ];

  for (const geneIdx of allGenes) {
    const slider = document.getElementById(`tp-target-${geneIdx}`) as HTMLInputElement;
    if (slider) {
      expression[geneIdx] = parseFloat(slider.value);
    }
  }

  return expression;
}

// Apply preset to modal sliders
function applyPresetToModal(presetName: string): void {
  const expression = generateExpressionFromPreset(presetName);
  populateTimepointModalSliders(expression);
}

// Helper to create slider row for modal
function createModalSliderRow(geneIdx: number, defaultVal: number): HTMLElement {
  const row = document.createElement('div');
  row.className = 'target-row';

  const label = document.createElement('label');
  label.textContent = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
  label.title = geneNames[geneIdx] ?? `Gene_${geneIdx}`;

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = `tp-target-${geneIdx}`;
  slider.min = '0';
  slider.max = '5';
  slider.step = '0.1';
  slider.value = String(defaultVal);

  const valSpan = document.createElement('span');
  valSpan.className = 'target-val';
  valSpan.id = `tp-val-target-${geneIdx}`;
  valSpan.textContent = defaultVal.toFixed(1);

  slider.addEventListener('input', () => {
    valSpan.textContent = parseFloat(slider.value).toFixed(1);
  });

  row.appendChild(label);
  row.appendChild(slider);
  row.appendChild(valSpan);
  return row;
}

// Regenerate modal sliders based on currently selected genes
function regenerateTimepointModalSliders(): void {
  const progContainer = document.getElementById('tp-prog-targets');
  const linContainer = document.getElementById('tp-lin-targets');
  const ligandContainer = document.getElementById('tp-ligand-targets');
  const receptorContainer = document.getElementById('tp-receptor-targets');

  // Clear existing sliders
  [progContainer, linContainer, ligandContainer, receptorContainer].forEach(c => {
    if (c) while (c.firstChild) c.removeChild(c.firstChild);
  });

  // Get selected genes (weight > 0)
  const selectedGenes = new Set<number>();
  for (let i = 0; i < nGenes; i++) {
    if (targetWeights[i] > 0) selectedGenes.add(i);
  }

  // Helper to add genes from a group that are selected
  function addSelectedFromGroup(container: HTMLElement | null, geneIndices: number[], defaultVal: number): number {
    if (!container) return 0;
    let count = 0;
    for (const geneIdx of geneIndices) {
      if (selectedGenes.has(geneIdx)) {
        container.appendChild(createModalSliderRow(geneIdx, defaultVal));
        count++;
      }
    }
    return count;
  }

  // Add sliders for selected genes in each group
  const progCount = addSelectedFromGroup(progContainer, geneGroups.tf_prog, 1.5);
  const linCount = addSelectedFromGroup(linContainer, geneGroups.tf_lin, 0.5);
  const ligandCount = addSelectedFromGroup(ligandContainer, geneGroups.ligand, 0.5);
  const receptorCount = addSelectedFromGroup(receptorContainer, geneGroups.receptor, 0.5);

  // Hide empty sections
  const sections = [
    { container: 'tp-prog-targets', count: progCount },
    { container: 'tp-lin-targets', count: linCount },
    { container: 'tp-ligand-targets', count: ligandCount },
    { container: 'tp-receptor-targets', count: receptorCount },
  ];

  for (const { container, count } of sections) {
    const el = document.getElementById(container);
    const section = el?.closest('.target-section') as HTMLElement;
    if (section) {
      section.style.display = count > 0 ? 'block' : 'none';
    }
  }
}

// Initialize timepoint modal (event handlers only, sliders generated on open)
function initializeTimepointModalSliders(): void {
  // Set up accordion headers in modal
  const modalBody = document.querySelector('.timepoint-modal-body');
  if (modalBody) {
    const headers = modalBody.querySelectorAll('.accordion-header');
    headers.forEach(header => {
      header.addEventListener('click', () => {
        const targetId = header.getAttribute('data-target');
        if (!targetId) return;
        const content = document.getElementById(targetId);
        const icon = header.querySelector('.accordion-icon');
        if (content && icon) {
          content.classList.toggle('collapsed');
          icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
        }
      });
    });
  }

  // Set up preset buttons in modal
  const presetGrid = document.getElementById('timepoint-preset-grid');
  if (presetGrid) {
    const buttons = presetGrid.querySelectorAll('.state-preset-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const presetName = btn.getAttribute('data-preset');
        if (presetName) applyPresetToModal(presetName);
      });
    });
  }
}

// Initialize timepoint modal event handlers
function initializeTimepointModal(): void {
  // Close button
  const closeBtn = document.getElementById('timepoint-modal-close');
  closeBtn?.addEventListener('click', closeTimepointModal);

  // Cancel button
  const cancelBtn = document.getElementById('btn-timepoint-cancel');
  cancelBtn?.addEventListener('click', closeTimepointModal);

  // Save button
  const saveBtn = document.getElementById('btn-timepoint-save');
  saveBtn?.addEventListener('click', saveTimepointModal);

  // Click outside to close
  const modal = document.getElementById('timepoint-modal');
  modal?.addEventListener('click', (e) => {
    if (e.target === modal) closeTimepointModal();
  });

  // Initialize sliders (after gene groups are populated)
  initializeTimepointModalSliders();
}

// Initialize target expression sliders
function initializeTargetSliders(): void {
  const groups: { id: string; indices: number[]; label: string }[] = [
    { id: 'prog-targets', indices: geneGroups.tf_prog, label: 'Progenitor TFs' },
    { id: 'lin-targets', indices: geneGroups.tf_lin, label: 'Lineage TFs' },
    { id: 'ligand-targets', indices: geneGroups.ligand, label: 'Ligands' },
    { id: 'receptor-targets', indices: geneGroups.receptor, label: 'Receptors' },
  ];

  for (const group of groups) {
    const container = document.getElementById(group.id);
    if (!container) continue;

    // Clear existing content safely
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    for (const idx of group.indices) {
      const name = geneNames[idx] ?? `Gene_${idx}`;
      const shortName = name.replace('TF_PROG_', 'P').replace('TF_LIN', 'L').replace('LIG_', 'Lig').replace('REC_', 'Rec');

      const row = document.createElement('div');
      row.className = 'target-row';

      const label = document.createElement('label');
      label.title = name;
      label.textContent = shortName;

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.id = `target-${idx}`;
      slider.min = '0';
      slider.max = '6';
      slider.step = '0.1';
      slider.value = targetExpression[idx].toFixed(1);

      const valSpan = document.createElement('span');
      valSpan.className = 'target-val';
      valSpan.id = `val-target-${idx}`;
      valSpan.textContent = targetExpression[idx].toFixed(1);

      row.appendChild(label);
      row.appendChild(slider);
      row.appendChild(valSpan);
      container.appendChild(row);

      // Event listener
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        targetExpression[idx] = val;
        valSpan.textContent = val.toFixed(1);
        // Clear active preset when manually adjusting
        activeStatePreset = null;
        updateStatePresetButtons();
        updateCharts();
      });
    }
  }
}

// Update weights based on preset
function updateWeights(): void {
  switch (weightPreset) {
    case 'uniform':
      targetWeights.fill(1);
      break;
    case 'tfs-only':
      targetWeights.fill(0);
      geneGroups.tf_prog.forEach(i => { targetWeights[i] = 2; });
      geneGroups.tf_lin.forEach(i => { targetWeights[i] = 2; });
      break;
    case 'select':
      // Only selected genes have weight=1, others have weight=0
      targetWeights.fill(0);
      for (const idx of selectedGenes) {
        targetWeights[idx] = 1;
      }
      break;
    case 'custom':
      // Read from custom weight sliders
      const progWeight = parseFloat((document.getElementById('weight-prog') as HTMLInputElement).value);
      const linWeight = parseFloat((document.getElementById('weight-lin') as HTMLInputElement).value);
      const ligandWeight = parseFloat((document.getElementById('weight-ligand') as HTMLInputElement).value);
      const receptorWeight = parseFloat((document.getElementById('weight-receptor') as HTMLInputElement).value);

      targetWeights.fill(0.1);
      geneGroups.tf_prog.forEach(i => { targetWeights[i] = progWeight; });
      geneGroups.tf_lin.forEach(i => { targetWeights[i] = linWeight; });
      geneGroups.ligand.forEach(i => { targetWeights[i] = ligandWeight; });
      geneGroups.receptor.forEach(i => { targetWeights[i] = receptorWeight; });
      break;
  }
}

// Update selected gene count display
function updateSelectedCount(): void {
  const countSpan = document.getElementById('selected-gene-count');
  if (countSpan) {
    countSpan.textContent = selectedGenes.size.toString();
  }
}

// Toggle gene selection
function toggleGeneSelection(idx: number, chip: HTMLElement): void {
  if (selectedGenes.has(idx)) {
    selectedGenes.delete(idx);
    chip.classList.remove('selected');
  } else {
    selectedGenes.add(idx);
    chip.classList.add('selected');
  }
  updateSelectedCount();
  if (weightPreset === 'select') {
    updateWeights();
  }
}

// Initialize gene selection UI
function initializeGeneSelectionUI(): void {
  const groups: { containerId: string; groupKey: keyof GeneGroups; label: string }[] = [
    { containerId: 'select-prog', groupKey: 'tf_prog', label: 'prog' },
    { containerId: 'select-lin', groupKey: 'tf_lin', label: 'lin' },
    { containerId: 'select-ligand', groupKey: 'ligand', label: 'ligand' },
    { containerId: 'select-receptor', groupKey: 'receptor', label: 'receptor' },
  ];

  for (const { containerId, groupKey, label } of groups) {
    const container = document.getElementById(containerId);
    if (!container) continue;

    // Clear existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const indices = geneGroups[groupKey];
    for (const idx of indices) {
      const name = geneNames[idx] ?? `Gene_${idx}`;
      const shortName = name
        .replace('TF_PROG_', 'P')
        .replace('TF_LIN', 'L')
        .replace('LIG_', '')
        .replace('REC_', '');

      const chip = document.createElement('button');
      chip.className = 'gene-chip';
      chip.textContent = shortName;
      chip.title = name;
      chip.dataset.geneIdx = idx.toString();

      // Check if already selected
      if (selectedGenes.has(idx)) {
        chip.classList.add('selected');
      }

      chip.addEventListener('click', () => {
        toggleGeneSelection(idx, chip);
      });

      container.appendChild(chip);
    }
  }

  // Set up All/None buttons
  const allBtns = document.querySelectorAll('.select-all-btn');
  const noneBtns = document.querySelectorAll('.select-none-btn');

  allBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.getAttribute('data-group');
      const groupKey = getGroupKeyFromLabel(group);
      if (!groupKey) return;

      const indices = geneGroups[groupKey];
      for (const idx of indices) {
        selectedGenes.add(idx);
      }

      // Update chip visuals
      const containerId = `select-${group}`;
      const container = document.getElementById(containerId);
      container?.querySelectorAll('.gene-chip').forEach(chip => {
        chip.classList.add('selected');
      });

      updateSelectedCount();
      if (weightPreset === 'select') {
        updateWeights();
      }
    });
  });

  noneBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.getAttribute('data-group');
      const groupKey = getGroupKeyFromLabel(group);
      if (!groupKey) return;

      const indices = geneGroups[groupKey];
      for (const idx of indices) {
        selectedGenes.delete(idx);
      }

      // Update chip visuals
      const containerId = `select-${group}`;
      const container = document.getElementById(containerId);
      container?.querySelectorAll('.gene-chip').forEach(chip => {
        chip.classList.remove('selected');
      });

      updateSelectedCount();
      if (weightPreset === 'select') {
        updateWeights();
      }
    });
  });

  updateSelectedCount();
}

// Helper to map group label to geneGroups key
function getGroupKeyFromLabel(label: string | null): keyof GeneGroups | null {
  switch (label) {
    case 'prog': return 'tf_prog';
    case 'lin': return 'tf_lin';
    case 'ligand': return 'ligand';
    case 'receptor': return 'receptor';
    default: return null;
  }
}

// Initialize weight controls
function initializeWeightControls(): void {
  const presetBtns = document.querySelectorAll('.preset-btn');
  const customPanel = document.getElementById('weight-custom');
  const selectPanel = document.getElementById('weight-select');

  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      weightPreset = btn.getAttribute('data-preset') as 'uniform' | 'tfs-only' | 'select' | 'custom';

      // Show/hide panels based on preset
      if (customPanel) {
        customPanel.classList.toggle('hidden', weightPreset !== 'custom');
      }
      if (selectPanel) {
        selectPanel.classList.toggle('hidden', weightPreset !== 'select');
      }

      updateWeights();
    });
  });

  // Custom weight sliders
  const weightSliders = ['weight-prog', 'weight-lin', 'weight-ligand', 'weight-receptor'];
  for (const id of weightSliders) {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valSpan = document.getElementById(`val-${id}`) as HTMLSpanElement;

    if (slider && valSpan) {
      slider.addEventListener('input', () => {
        valSpan.textContent = parseFloat(slider.value).toFixed(1);
        if (weightPreset === 'custom') {
          updateWeights();
        }
      });
    }
  }
}

// Initialize accordion toggles
function initializeAccordions(): void {
  const headers = document.querySelectorAll('.accordion-header');
  headers.forEach(header => {
    header.addEventListener('click', () => {
      const targetId = header.getAttribute('data-target');
      if (!targetId) return;

      const content = document.getElementById(targetId);
      if (!content) return;

      const icon = header.querySelector('.accordion-icon');
      content.classList.toggle('collapsed');

      if (icon) {
        icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
      }
    });
  });
}

// Initialize canvases
function initializeCanvases(): void {
  fitnessCanvas = document.getElementById('fitness-chart') as HTMLCanvasElement;
  comparisonCanvas = document.getElementById('comparison-chart') as HTMLCanvasElement;
  scatterCanvas = document.getElementById('scatter-chart') as HTMLCanvasElement;
  errorCanvas = document.getElementById('error-chart') as HTMLCanvasElement;
  sensitivityCanvas = document.getElementById('sensitivity-chart') as HTMLCanvasElement;

  if (fitnessCanvas) fitnessCtx = fitnessCanvas.getContext('2d')!;
  if (comparisonCanvas) comparisonCtx = comparisonCanvas.getContext('2d')!;
  if (scatterCanvas) scatterCtx = scatterCanvas.getContext('2d')!;
  if (errorCanvas) errorCtx = errorCanvas.getContext('2d')!;
  if (sensitivityCanvas) sensitivityCtx = sensitivityCanvas.getContext('2d')!;

  // Handle resize
  window.addEventListener('resize', resizeCanvases);
  resizeCanvases();
}

function resizeCanvases(): void {
  const canvases = [fitnessCanvas, comparisonCanvas, scatterCanvas, errorCanvas, sensitivityCanvas];
  for (const canvas of canvases) {
    if (!canvas) continue;
    const rect = canvas.parentElement?.getBoundingClientRect();
    if (rect && rect.width > 0 && rect.height > 0) {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Reset transform before applying new scale
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
      }
    }
  }
  updateCharts();
}

// Draw fitness history chart
function drawFitnessChart(): void {
  if (!fitnessCtx || !fitnessCanvas) return;

  const ctx = fitnessCtx;
  const dpr = window.devicePixelRatio || 1;
  const w = fitnessCanvas.width / dpr;
  const h = fitnessCanvas.height / dpr;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  if (fitnessHistory.length === 0) {
    ctx.fillStyle = '#6e7681';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for optimization...', w / 2, h / 2);
    return;
  }

  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Find ranges
  const maxGen = Math.max(...fitnessHistory.map(d => d.gen));
  const maxFitness = Math.max(...fitnessHistory.map(d => Math.max(d.fitness, d.meanFitness))) * 1.1;
  const minFitness = 0;

  // Draw axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, h - padding.bottom);
  ctx.lineTo(w - padding.right, h - padding.bottom);
  ctx.stroke();

  // Draw grid lines
  ctx.strokeStyle = '#21262d';
  ctx.setLineDash([2, 4]);
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Draw mean fitness line
  ctx.strokeStyle = '#8b949e';
  ctx.lineWidth = 1;
  ctx.beginPath();
  fitnessHistory.forEach((d, i) => {
    const x = padding.left + (d.gen / maxGen) * plotW;
    const y = h - padding.bottom - ((d.meanFitness - minFitness) / (maxFitness - minFitness)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw best fitness line
  ctx.strokeStyle = '#39c5cf';
  ctx.lineWidth = 2;
  ctx.beginPath();
  fitnessHistory.forEach((d, i) => {
    const x = padding.left + (d.gen / maxGen) * plotW;
    const y = h - padding.bottom - ((d.fitness - minFitness) / (maxFitness - minFitness)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Generation', padding.left + plotW / 2, h - 5);

  ctx.save();
  ctx.translate(15, padding.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Fitness', 0, 0);
  ctx.restore();

  // Current values
  if (fitnessHistory.length > 0) {
    const last = fitnessHistory[fitnessHistory.length - 1];
    ctx.fillStyle = '#39c5cf';
    ctx.textAlign = 'right';
    ctx.fillText(`Best: ${last.fitness.toFixed(4)}`, w - padding.right, padding.top + 15);
    ctx.fillStyle = '#8b949e';
    ctx.fillText(`Mean: ${last.meanFitness.toFixed(4)}`, w - padding.right, padding.top + 30);
  }
}

// Gene type colors
const geneColors = {
  tf_prog: '#f85149',
  tf_lin: '#58a6ff',
  ligand: '#3fb950',
  receptor: '#a371f7',
  target: '#ffa657',
  housekeeping: '#8b949e',
  other: '#6e7681',
};

// Draw expression comparison chart - grouped by gene type
function drawComparisonChart(): void {
  if (!comparisonCtx || !comparisonCanvas) return;

  const ctx = comparisonCtx;
  const dpr = window.devicePixelRatio || 1;
  const w = comparisonCanvas.width / dpr;
  const h = comparisonCanvas.height / dpr;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  const padding = { top: 25, right: 15, bottom: 45, left: 45 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Gene groups to display with their info
  const groups = [
    { key: 'tf_prog', label: 'Prog TFs', indices: geneGroups.tf_prog, color: geneColors.tf_prog },
    { key: 'tf_lin', label: 'Lin TFs', indices: geneGroups.tf_lin, color: geneColors.tf_lin },
    { key: 'ligand', label: 'Ligands', indices: geneGroups.ligand, color: geneColors.ligand },
    { key: 'receptor', label: 'Receptors', indices: geneGroups.receptor, color: geneColors.receptor },
  ];

  const totalGenes = groups.reduce((sum, g) => sum + g.indices.length, 0);
  const groupGap = 15;
  const totalGaps = (groups.length - 1) * groupGap;
  const barAreaWidth = plotW - totalGaps;
  const barWidth = Math.min(8, barAreaWidth / totalGenes / 2.2);
  const pairWidth = barWidth * 2.4;

  // Draw grid lines
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let v = 0; v <= 6; v += 2) {
    const y = padding.top + plotH - (v / 6) * plotH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  // Draw Y-axis labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let v = 0; v <= 6; v += 2) {
    const y = padding.top + plotH - (v / 6) * plotH;
    ctx.fillText(String(v), padding.left - 5, y + 3);
  }

  // Y-axis title
  ctx.save();
  ctx.translate(12, padding.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Expression Level', 0, 0);
  ctx.restore();

  let xOffset = padding.left;

  groups.forEach((group, gi) => {
    const groupWidth = group.indices.length * pairWidth;

    // Draw group background
    ctx.fillStyle = 'rgba(48, 54, 61, 0.3)';
    ctx.fillRect(xOffset - 2, padding.top, groupWidth + 4, plotH);

    // Draw group label
    ctx.fillStyle = group.color;
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(group.label, xOffset + groupWidth / 2, h - padding.bottom + 30);

    // Draw bars for each gene
    group.indices.forEach((idx, i) => {
      const x = xOffset + i * pairWidth;
      const targetVal = targetExpression[idx] ?? 0;
      const simVal = bestSimulatedExpression ? (bestSimulatedExpression[idx] ?? 0) : 0;
      const isWeighted = targetWeights[idx] > 0;

      // Target bar (left, purple-ish)
      const targetH = (targetVal / 6) * plotH;
      ctx.fillStyle = isWeighted ? '#a371f7' : 'rgba(163, 113, 247, 0.3)';
      ctx.fillRect(x, padding.top + plotH - targetH, barWidth, targetH);

      // Simulated bar (right, green-ish)
      if (bestSimulatedExpression) {
        const simH = (simVal / 6) * plotH;
        ctx.fillStyle = isWeighted ? '#3fb950' : 'rgba(63, 185, 80, 0.3)';
        ctx.fillRect(x + barWidth * 1.2, padding.top + plotH - simH, barWidth, simH);
      }

      // Draw gene label for every Nth gene
      const labelInterval = Math.max(1, Math.floor(group.indices.length / 4));
      if (i % labelInterval === 0) {
        const name = geneNames[idx]?.replace('TF_PROG_', 'P').replace('TF_LIN', 'L').replace('LIG_', '').replace('REC_', '') ?? '';
        ctx.fillStyle = '#6e7681';
        ctx.font = '8px sans-serif';
        ctx.save();
        ctx.translate(x + barWidth, padding.top + plotH + 5);
        ctx.rotate(Math.PI / 3);
        ctx.textAlign = 'left';
        ctx.fillText(name, 0, 0);
        ctx.restore();
      }
    });

    xOffset += groupWidth + groupGap;
  });

  // Draw axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.lineTo(w - padding.right, padding.top + plotH);
  ctx.stroke();
}

// Draw scatter plot (target vs simulated) with R² and better styling
function drawScatterChart(): void {
  if (!scatterCtx || !scatterCanvas) return;

  const ctx = scatterCtx;
  const dpr = window.devicePixelRatio || 1;
  const w = scatterCanvas.width / dpr;
  const h = scatterCanvas.height / dpr;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  const padding = { top: 20, right: 15, bottom: 35, left: 40 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;
  const maxVal = 6;

  // Draw grid
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let v = 0; v <= 6; v += 2) {
    const pos = (v / maxVal);
    // Horizontal
    const y = padding.top + plotH * (1 - pos);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
    // Vertical
    const x = padding.left + plotW * pos;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, padding.top + plotH);
    ctx.stroke();
  }

  // Draw identity line (y = x, where target = simulated = perfect match)
  // Use a subtle solid line, not dashed, so it's clearly a reference
  ctx.strokeStyle = 'rgba(139, 148, 158, 0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top + plotH);
  ctx.lineTo(padding.left + plotW, padding.top);
  ctx.stroke();

  // Label for the identity line (only when there's data)
  if (bestSimulatedExpression) {
    ctx.fillStyle = 'rgba(139, 148, 158, 0.5)';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'left';
    ctx.save();
    ctx.translate(padding.left + plotW - 20, padding.top + 18);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText('y=x', 0, 0);
    ctx.restore();
  }

  // Draw axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.lineTo(w - padding.right, padding.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  for (let v = 0; v <= 6; v += 2) {
    const x = padding.left + (v / maxVal) * plotW;
    ctx.fillText(String(v), x, h - padding.bottom + 12);
  }
  ctx.textAlign = 'right';
  for (let v = 0; v <= 6; v += 2) {
    const y = padding.top + plotH * (1 - v / maxVal);
    ctx.fillText(String(v), padding.left - 5, y + 3);
  }

  // Axis titles
  ctx.fillStyle = '#6e7681';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Target', padding.left + plotW / 2, h - 3);
  ctx.save();
  ctx.translate(10, padding.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Simulated', 0, 0);
  ctx.restore();

  if (!bestSimulatedExpression) {
    ctx.fillStyle = '#6e7681';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for optimization...', w / 2, h / 2);
    return;
  }

  // Collect points data
  const allGenes = [...geneGroups.tf_prog, ...geneGroups.tf_lin, ...geneGroups.ligand, ...geneGroups.receptor];
  const points: { x: number; y: number; color: string; weighted: boolean; idx: number }[] = [];

  for (const idx of allGenes) {
    const target = targetExpression[idx] ?? 0;
    const simulated = bestSimulatedExpression[idx] ?? 0;
    const isWeighted = targetWeights[idx] > 0;

    let color = geneColors.other;
    if (geneGroups.tf_prog.includes(idx)) color = geneColors.tf_prog;
    else if (geneGroups.tf_lin.includes(idx)) color = geneColors.tf_lin;
    else if (geneGroups.ligand.includes(idx)) color = geneColors.ligand;
    else if (geneGroups.receptor.includes(idx)) color = geneColors.receptor;

    points.push({
      x: padding.left + (target / maxVal) * plotW,
      y: padding.top + plotH * (1 - simulated / maxVal),
      color,
      weighted: isWeighted,
      idx,
    });
  }

  // Draw unweighted points first (faded)
  points.filter(p => !p.weighted).forEach(p => {
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // Draw weighted points on top (solid with border)
  ctx.globalAlpha = 1;
  points.filter(p => p.weighted).forEach(p => {
    ctx.fillStyle = p.color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  // Compute and display R² for weighted genes
  const weightedPoints = points.filter(p => p.weighted);
  if (weightedPoints.length > 1) {
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    const n = weightedPoints.length;

    weightedPoints.forEach(p => {
      const tx = targetExpression[p.idx] ?? 0;
      const sy = bestSimulatedExpression![p.idx] ?? 0;
      sumX += tx;
      sumY += sy;
      sumXY += tx * sy;
      sumX2 += tx * tx;
      sumY2 += sy * sy;
    });

    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    const r = den > 0 ? num / den : 0;
    const r2 = r * r;

    // Display R²
    ctx.fillStyle = '#39c5cf';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`R² = ${r2.toFixed(3)}`, w - padding.right - 5, padding.top + 15);
  }
}

// Draw error chart - horizontal bars showing top mismatches
function drawErrorChart(): void {
  if (!errorCtx || !errorCanvas) return;

  const ctx = errorCtx;
  const dpr = window.devicePixelRatio || 1;
  const w = errorCanvas.width / dpr;
  const h = errorCanvas.height / dpr;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  const padding = { top: 10, right: 40, bottom: 20, left: 60 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  if (!bestSimulatedExpression) {
    ctx.fillStyle = '#6e7681';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for results...', w / 2, h / 2);
    return;
  }

  // Calculate errors for all displayed gene groups (show all if no weights selected)
  const allGenes = [...geneGroups.tf_prog, ...geneGroups.tf_lin, ...geneGroups.ligand, ...geneGroups.receptor];
  const errors: { idx: number; error: number; target: number; sim: number; name: string; weighted: boolean }[] = [];

  // Check if any genes are weighted
  const hasWeightedGenes = allGenes.some(idx => (targetWeights[idx] ?? 0) > 0);

  for (const idx of allGenes) {
    const weight = targetWeights[idx] ?? 0;
    // If no weighted genes, show all; otherwise only show weighted
    if (hasWeightedGenes && weight === 0) continue;

    const target = targetExpression[idx] ?? 0;
    const sim = bestSimulatedExpression[idx] ?? 0;
    const error = sim - target;  // Positive = overshot, negative = undershot

    let name = geneNames[idx] ?? `Gene${idx}`;
    name = name.replace('TF_PROG_', 'P').replace('TF_LIN', 'L').replace('LIG_', 'Lig').replace('REC_', 'Rec');

    errors.push({ idx, error, target, sim, name, weighted: weight > 0 });
  }

  // Sort by absolute error and take top N
  errors.sort((a, b) => Math.abs(b.error) - Math.abs(a.error));
  const topN = Math.min(8, errors.length);
  const displayErrors = errors.slice(0, topN);

  if (displayErrors.length === 0) {
    ctx.fillStyle = '#6e7681';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No genes to display', w / 2, h / 2);
    return;
  }

  const totalHeight = displayErrors.length * 20;
  const barHeight = Math.min(14, Math.max(10, (plotH - 5) / displayErrors.length - 2));
  const barGap = Math.max(2, (plotH - displayErrors.length * barHeight) / (displayErrors.length + 1));
  const maxError = Math.max(...displayErrors.map(e => Math.abs(e.error)), 0.5);

  // Draw center line (zero error)
  const centerX = padding.left + plotW / 2;
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(centerX, padding.top);
  ctx.lineTo(centerX, padding.top + plotH);
  ctx.stroke();

  // Draw error bars
  displayErrors.forEach((e, i) => {
    const y = padding.top + barGap + i * (barHeight + barGap);
    const barW = Math.max(2, (Math.abs(e.error) / maxError) * (plotW / 2 - 5));

    // Gene name label
    ctx.fillStyle = e.weighted ? '#c9d1d9' : '#6e7681';
    ctx.font = `${e.weighted ? 'bold ' : ''}9px sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(e.name, padding.left - 4, y + barHeight / 2 + 3);

    // Error bar
    const barColor = e.error > 0 ? '#ffa657' : '#58a6ff';
    ctx.fillStyle = barColor;
    if (e.error > 0) {
      // Overshot (right side, orange)
      ctx.fillRect(centerX + 1, y, barW, barHeight);
    } else if (e.error < 0) {
      // Undershot (left side, blue)
      ctx.fillRect(centerX - barW - 1, y, barW, barHeight);
    } else {
      // Zero error - small mark at center
      ctx.fillStyle = '#3fb950';
      ctx.fillRect(centerX - 1, y, 2, barHeight);
    }

    // Error value on the outside
    ctx.fillStyle = e.error === 0 ? '#3fb950' : barColor;
    ctx.font = '8px sans-serif';
    if (e.error >= 0) {
      ctx.textAlign = 'left';
      ctx.fillText((e.error >= 0 ? '+' : '') + e.error.toFixed(2), centerX + barW + 4, y + barHeight / 2 + 3);
    } else {
      ctx.textAlign = 'right';
      ctx.fillText(e.error.toFixed(2), centerX - barW - 4, y + barHeight / 2 + 3);
    }
  });

  // Labels at bottom
  ctx.fillStyle = '#6e7681';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Under', padding.left + plotW * 0.22, h - 5);
  ctx.fillText('Over', padding.left + plotW * 0.78, h - 5);

  // Arrows
  ctx.fillText('←', padding.left + plotW * 0.08, h - 5);
  ctx.fillText('→', padding.left + plotW * 0.92, h - 5);
}

// Update all charts
function updateCharts(): void {
  drawFitnessChart();
  drawComparisonChart();
  drawScatterChart();
  drawErrorChart();
  drawSensitivityChart();
}

// Sensitivity Analysis Functions

// Run sensitivity analysis on current best parameters
function runSensitivityAnalysis(): void {
  if (!bestParams) return;

  const btn = document.getElementById('btn-sensitivity') as HTMLButtonElement;
  const content = document.getElementById('sensitivity-content');

  // Show running state
  btn.classList.add('running');
  btn.textContent = 'Analyzing...';
  btn.disabled = true;

  // Get current encoding config
  const includeGlobal = (document.getElementById('opt-global') as HTMLInputElement).checked;
  const includeKnockouts = (document.getElementById('opt-knockouts') as HTMLInputElement).checked;
  const includeMorphogens = (document.getElementById('opt-morphogens') as HTMLInputElement).checked;
  const includeModifiers = (document.getElementById('opt-modifiers') as HTMLInputElement).checked;

  const config: EncodingConfig = {
    includeGlobal,
    includeKnockouts,
    includeMorphogens,
    includeModifiers,
    geneNames,
    knockoutGeneIndices: knockoutCandidates,
    modifierGeneIndices: knockoutCandidates,
  };

  // Create target state for fitness evaluation
  const targetState: TargetState = {
    expression: targetExpression,
    weights: targetWeights,
  };

  // Encode current best params
  const baseVector = encode(bestParams, config);
  const dims = getDimensions(config);
  const paramNames = getParameterNames(config);

  // Compute base fitness
  const simTime = parseInt((document.getElementById('sim-time') as HTMLInputElement).value);
  const baseSim = engine.runSimulation(bestParams, Math.PI, simTime);
  const baseFitness = computeFitness(baseSim, targetState);

  // Compute sensitivity for each parameter
  const delta = 0.05;  // Perturbation size (normalized)
  sensitivityResults = [];

  for (let i = 0; i < dims; i++) {
    // Perturb parameter
    const perturbedVector = new Float64Array(baseVector);
    perturbedVector[i] = Math.min(1, Math.max(0, perturbedVector[i] + delta));

    // Decode and simulate
    const perturbedParams = decode(perturbedVector, config);
    const perturbedSim = engine.runSimulation(perturbedParams, Math.PI, simTime);
    const perturbedFitness = computeFitness(perturbedSim, targetState);

    // Compute sensitivity (absolute change per unit perturbation)
    const sensitivity = Math.abs(perturbedFitness - baseFitness) / delta;

    sensitivityResults.push({
      paramName: paramNames[i] ?? `Param_${i}`,
      sensitivity,
    });
  }

  // Sort by sensitivity (highest first)
  sensitivityResults.sort((a, b) => b.sensitivity - a.sensitivity);

  // Update display
  updateSensitivityDisplay();
  drawSensitivityChart();

  // Reset button state
  btn.classList.remove('running');
  btn.textContent = 'Run Analysis';
  btn.disabled = false;
}

// Update sensitivity display text
function updateSensitivityDisplay(): void {
  const content = document.getElementById('sensitivity-content');
  if (!content) return;

  // Clear existing content
  while (content.firstChild) {
    content.removeChild(content.firstChild);
  }

  if (sensitivityResults.length === 0) {
    const p = document.createElement('p');
    p.className = 'placeholder';
    p.textContent = 'Run optimization first';
    content.appendChild(p);
    return;
  }

  // Show top 5 most sensitive parameters
  const maxSens = sensitivityResults[0]?.sensitivity ?? 1;
  const top5 = sensitivityResults.slice(0, 5);

  for (const result of top5) {
    const row = document.createElement('div');
    row.className = 'sensitivity-result';

    const name = document.createElement('span');
    name.className = 'param-name';
    // Shorten long names
    let displayName = result.paramName
      .replace('Morphogen_', 'M')
      .replace('Mod_TF_PROG_', 'Mod_P')
      .replace('Mod_TF_LIN', 'Mod_L')
      .replace('KO_TF_PROG_', 'KO_P')
      .replace('KO_TF_LIN', 'KO_L');
    if (displayName.length > 12) {
      displayName = displayName.substring(0, 11) + '…';
    }
    name.textContent = displayName;
    name.title = result.paramName;

    const barContainer = document.createElement('div');
    barContainer.className = 'sensitivity-bar';
    const barFill = document.createElement('div');
    barFill.className = 'sensitivity-bar-fill';
    barFill.style.width = `${(result.sensitivity / maxSens) * 100}%`;
    barContainer.appendChild(barFill);

    const value = document.createElement('span');
    value.className = 'sensitivity-value';
    value.textContent = result.sensitivity.toFixed(2);

    row.appendChild(name);
    row.appendChild(barContainer);
    row.appendChild(value);
    content.appendChild(row);
  }
}

// Draw sensitivity bar chart
function drawSensitivityChart(): void {
  if (!sensitivityCtx || !sensitivityCanvas) return;

  const ctx = sensitivityCtx;
  const dpr = window.devicePixelRatio || 1;
  const w = sensitivityCanvas.width / dpr;
  const h = sensitivityCanvas.height / dpr;

  // Clear
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  if (sensitivityResults.length === 0) {
    ctx.fillStyle = '#6e7681';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Run analysis to see results', w / 2, h / 2);
    return;
  }

  const padding = { top: 10, right: 10, bottom: 25, left: 10 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Show top 10 parameters
  const top10 = sensitivityResults.slice(0, 10);
  const maxSens = top10[0]?.sensitivity ?? 1;
  const barWidth = (plotW - (top10.length - 1) * 2) / top10.length;

  // Draw bars
  top10.forEach((result, i) => {
    const x = padding.left + i * (barWidth + 2);
    const barH = (result.sensitivity / maxSens) * plotH;
    const y = padding.top + plotH - barH;

    // Color based on parameter type
    let color = '#39c5cf';  // Default cyan
    if (result.paramName.startsWith('Mod_')) color = '#a371f7';  // Purple for modifiers
    else if (result.paramName.startsWith('KO_')) color = '#f85149';  // Red for knockouts
    else if (result.paramName.startsWith('Morphogen')) color = '#3fb950';  // Green for morphogens

    ctx.fillStyle = color;
    ctx.fillRect(x, y, barWidth, barH);
  });

  // Draw x-axis
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top + plotH);
  ctx.lineTo(w - padding.right, padding.top + plotH);
  ctx.stroke();

  // Draw abbreviated labels
  ctx.fillStyle = '#6e7681';
  ctx.font = '7px sans-serif';
  ctx.textAlign = 'center';

  top10.forEach((result, i) => {
    const x = padding.left + i * (barWidth + 2) + barWidth / 2;
    // Show very abbreviated name
    let label = result.paramName
      .replace('Hill N', 'HN')
      .replace('Hill K', 'HK')
      .replace('Prog Bias', 'PB')
      .replace('Lin Bias', 'LB')
      .replace('Prog Decay', 'PD')
      .replace('Lin Decay', 'LD')
      .replace('Inhibition Mult', 'IM')
      .replace('Morphogen Time', 'MT')
      .replace('Morphogen Strength', 'MS')
      .replace('Init Prog T F', 'IP')
      .replace('Init Lin T F', 'IL')
      .replace('Ligand Bias', 'LiB')
      .replace('Receptor Bias', 'RB')
      .replace('Morphogen_Lin', 'M')
      .replace('Mod_TF_PROG_', 'P')
      .replace('Mod_TF_LIN', 'L')
      .replace('KO_TF_PROG_', 'P')
      .replace('KO_TF_LIN', 'L');

    if (label.length > 4) label = label.substring(0, 4);
    ctx.fillText(label, x, h - 5);
  });
}

// Stability Analysis Functions

// Run stability check on current best parameters
function runStabilityCheck(): void {
  if (!bestParams) return;

  const btn = document.getElementById('btn-stability') as HTMLButtonElement;
  const content = document.getElementById('stability-content');

  // Show running state
  btn.classList.add('running');
  btn.textContent = 'Checking...';
  btn.disabled = true;

  // Get threshold from slider
  const threshold = parseFloat((document.getElementById('stability-threshold') as HTMLInputElement).value);
  const simTime = parseInt((document.getElementById('sim-time') as HTMLInputElement).value);
  const uniformMorphogens = (document.getElementById('opt-uniform-morphogens') as HTMLInputElement).checked;

  // Run stability check
  stabilityResult = engine.checkStability(
    bestParams,
    simTime,
    simTime + 6,  // Extended time = simTime + 6 hours
    threshold,
    uniformMorphogens
  );

  // Update display
  updateStabilityDisplay();

  // Reset button state
  btn.classList.remove('running');
  btn.textContent = 'Check Stability';
  btn.disabled = false;
}

// Update stability display
function updateStabilityDisplay(): void {
  const content = document.getElementById('stability-content');
  const driftingDiv = document.getElementById('stability-drifting-genes');

  // Helper to clear element
  function clearElement(el: HTMLElement | null): void {
    if (!el) return;
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  clearElement(content);
  clearElement(driftingDiv);

  if (!stabilityResult) {
    if (content) {
      const p = document.createElement('p');
      p.className = 'placeholder';
      p.textContent = 'Run optimization first';
      content.appendChild(p);
    }
    return;
  }

  if (!content) return;

  // Status indicator
  const statusDiv = document.createElement('div');
  const statusClass = stabilityResult.isStable
    ? 'stable'
    : (stabilityResult.driftMagnitude < 0.2 ? 'transient' : 'unstable');
  statusDiv.className = `stability-status ${statusClass}`;

  const icon = document.createElement('span');
  icon.className = 'stability-icon';
  icon.textContent = stabilityResult.isStable ? '✓' : (stabilityResult.driftMagnitude < 0.2 ? '~' : '✗');

  const label = document.createElement('span');
  label.className = 'stability-label';
  label.textContent = stabilityResult.isStable
    ? 'STABLE'
    : (stabilityResult.driftMagnitude < 0.2 ? 'TRANSIENT' : 'UNSTABLE');

  statusDiv.appendChild(icon);
  statusDiv.appendChild(label);
  content.appendChild(statusDiv);

  // Metrics
  const metricsDiv = document.createElement('div');
  metricsDiv.className = 'stability-metrics';

  // Drift magnitude
  const driftMetric = document.createElement('div');
  driftMetric.className = 'stability-metric';
  const driftLabel = document.createElement('div');
  driftLabel.className = 'stability-metric-label';
  driftLabel.textContent = 'Drift';
  const driftValue = document.createElement('div');
  driftValue.className = 'stability-metric-value';
  driftValue.textContent = stabilityResult.driftMagnitude.toFixed(3);
  driftMetric.appendChild(driftLabel);
  driftMetric.appendChild(driftValue);

  // Equilibrium time
  const eqMetric = document.createElement('div');
  eqMetric.className = 'stability-metric';
  const eqLabel = document.createElement('div');
  eqLabel.className = 'stability-metric-label';
  eqLabel.textContent = 'Equilibrium';
  const eqValue = document.createElement('div');
  eqValue.className = 'stability-metric-value';
  eqValue.textContent = `${stabilityResult.timeToEquilibrium.toFixed(1)}h`;
  eqMetric.appendChild(eqLabel);
  eqMetric.appendChild(eqValue);

  metricsDiv.appendChild(driftMetric);
  metricsDiv.appendChild(eqMetric);
  content.appendChild(metricsDiv);

  // Drifting genes
  if (driftingDiv && stabilityResult.driftPerGene.size > 0) {
    const header = document.createElement('div');
    header.className = 'drifting-genes-header';
    header.textContent = 'Drifting genes:';
    driftingDiv.appendChild(header);

    // Sort by absolute drift
    const sortedDrift = Array.from(stabilityResult.driftPerGene.entries())
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 6);  // Show top 6

    for (const [geneIdx, drift] of sortedDrift) {
      const chip = document.createElement('span');
      chip.className = `drifting-gene ${drift > 0 ? 'positive' : 'negative'}`;
      const geneName = geneNames[geneIdx] ?? `Gene_${geneIdx}`;
      const shortName = geneName.length > 10 ? geneName.substring(0, 9) + '…' : geneName;
      chip.textContent = `${shortName}: ${drift > 0 ? '+' : ''}${drift.toFixed(2)}`;
      chip.title = `${geneName}: ${drift > 0 ? '+' : ''}${drift.toFixed(3)}`;
      driftingDiv.appendChild(chip);
    }
  }
}

// Initialize stability button
function initializeStabilityButton(): void {
  const btn = document.getElementById('btn-stability');
  btn?.addEventListener('click', runStabilityCheck);

  // Threshold slider
  const thresholdSlider = document.getElementById('stability-threshold') as HTMLInputElement;
  const thresholdVal = document.getElementById('stability-threshold-val');
  thresholdSlider?.addEventListener('input', () => {
    if (thresholdVal) {
      thresholdVal.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    }
  });
}

// Update results display - using safe DOM manipulation
function updateResults(): void {
  const paramsDiv = document.getElementById('result-params');
  const knockoutsDiv = document.getElementById('result-knockouts');
  const morphogensDiv = document.getElementById('result-morphogens');
  const modifiersDiv = document.getElementById('result-modifiers');

  // Helper to clear element
  function clearElement(el: HTMLElement | null): void {
    if (!el) return;
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  // Helper to add placeholder
  function addPlaceholder(el: HTMLElement | null, text: string): void {
    if (!el) return;
    const p = document.createElement('p');
    p.className = 'placeholder';
    p.textContent = text;
    el.appendChild(p);
  }

  if (!bestParams) {
    clearElement(paramsDiv);
    clearElement(knockoutsDiv);
    clearElement(morphogensDiv);
    clearElement(modifiersDiv);
    addPlaceholder(paramsDiv, 'Run optimization to see results');
    addPlaceholder(knockoutsDiv, '-');
    addPlaceholder(morphogensDiv, '-');
    addPlaceholder(modifiersDiv, '-');
    return;
  }

  // Global parameters
  if (paramsDiv) {
    clearElement(paramsDiv);
    for (const key of GLOBAL_PARAM_KEYS) {
      const val = bestParams.global[key];
      const name = key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());

      const row = document.createElement('div');
      row.className = 'param-row';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'param-name';
      nameSpan.textContent = name;

      const valSpan = document.createElement('span');
      valSpan.className = 'param-value';
      valSpan.textContent = val.toFixed(3);

      row.appendChild(nameSpan);
      row.appendChild(valSpan);
      paramsDiv.appendChild(row);
    }
  }

  // Knockouts
  if (knockoutsDiv) {
    clearElement(knockoutsDiv);
    if (bestParams.knockouts.size === 0) {
      addPlaceholder(knockoutsDiv, 'None');
    } else {
      for (const idx of bestParams.knockouts) {
        const item = document.createElement('span');
        item.className = 'knockout-item';
        item.textContent = geneNames[idx] ?? `Gene_${idx}`;
        knockoutsDiv.appendChild(item);
      }
    }
  }

  // Morphogens
  if (morphogensDiv) {
    clearElement(morphogensDiv);
    bestParams.morphogens.forEach((m, i) => {
      const item = document.createElement('span');
      item.className = 'morphogen-item';
      item.textContent = `Lin${i + 1}: ${m.strength.toFixed(2)}`;
      morphogensDiv.appendChild(item);
    });
  }

  // Gene Modifiers
  if (modifiersDiv) {
    clearElement(modifiersDiv);
    if (bestParams.geneModifiers.size === 0) {
      addPlaceholder(modifiersDiv, 'None');
    } else {
      // Sort modifiers by effect (inhibited first, then overexpressed)
      const sortedModifiers = Array.from(bestParams.geneModifiers.entries())
        .sort((a, b) => a[1] - b[1]);

      for (const [idx, modifier] of sortedModifiers) {
        const item = document.createElement('span');
        // Color-code by effect type
        if (modifier < 0.5) {
          item.className = 'modifier-item inhibited';
        } else if (modifier > 1.5) {
          item.className = 'modifier-item overexpressed';
        } else {
          item.className = 'modifier-item';
        }
        const geneName = geneNames[idx] ?? `Gene_${idx}`;
        const shortName = geneName.replace('TF_PROG_', 'P').replace('TF_LIN', 'L').replace('LIG_', 'Lig').replace('REC_', 'Rec');
        item.textContent = `${shortName}: ${modifier.toFixed(2)}`;
        item.title = geneName;
        modifiersDiv.appendChild(item);
      }
    }
  }

  // Enable action buttons
  (document.getElementById('btn-apply') as HTMLButtonElement).disabled = false;
  (document.getElementById('btn-export') as HTMLButtonElement).disabled = false;
  (document.getElementById('btn-sensitivity') as HTMLButtonElement).disabled = false;
  (document.getElementById('btn-stability') as HTMLButtonElement).disabled = false;

  // Auto-run stability check
  runStabilityCheck();
}

// Start optimization
function startOptimization(): void {
  if (isRunning) return;

  // Get config from UI
  const includeGlobal = (document.getElementById('opt-global') as HTMLInputElement).checked;
  const includeKnockouts = (document.getElementById('opt-knockouts') as HTMLInputElement).checked;
  const includeMorphogens = (document.getElementById('opt-morphogens') as HTMLInputElement).checked;
  const includeModifiers = (document.getElementById('opt-modifiers') as HTMLInputElement).checked;
  const minimalIntervention = (document.getElementById('opt-minimal-intervention') as HTMLInputElement).checked;
  const interventionPenalty = parseFloat((document.getElementById('intervention-penalty') as HTMLInputElement).value);
  const requireHomogeneous = (document.getElementById('opt-homogeneous') as HTMLInputElement).checked;
  const maxGenerations = parseInt((document.getElementById('max-generations') as HTMLInputElement).value);
  const populationSize = parseInt((document.getElementById('pop-size') as HTMLInputElement).value);
  const simTime = parseInt((document.getElementById('sim-time') as HTMLInputElement).value);
  const targetLineage = parseInt((document.getElementById('target-lineage') as HTMLSelectElement).value);
  const uniformMorphogens = (document.getElementById('opt-uniform-morphogens') as HTMLInputElement).checked;

  // Update weights
  updateWeights();

  // Reset state
  fitnessHistory = [];
  bestSimulatedExpression = null;
  bestParams = null;

  // Create worker
  worker = new Worker(new URL('./inference/worker.ts', import.meta.url), { type: 'module' });

  worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
    const msg = event.data;

    switch (msg.type) {
      case 'progress':
        handleProgress(msg);
        break;
      case 'done':
        handleDone(msg);
        break;
      case 'error':
        console.error('Worker error:', msg.error);
        stopOptimization();
        break;
    }
  };

  worker.onerror = (error) => {
    console.error('Worker error:', error);
    stopOptimization();
  };

  // Prepare time-course targets if in time-course mode
  const timeCourseData: TimePointTarget[] = timeCourseMode
    ? timeCourseTargets.map(tp => ({
        time: tp.time,
        expression: tp.expression,
        weight: tp.weight,
      }))
    : [];

  // Send start message
  const startMsg: WorkerStartMessage = {
    type: 'start',
    grnData: data,
    target: {
      expression: Array.from(targetExpression),
      weights: Array.from(targetWeights),
    },
    timeCourseTarget: timeCourseData,  // Add time-course targets
    config: {
      includeGlobal,
      includeKnockouts,
      includeMorphogens,
      includeModifiers,
      minimalIntervention,  // Enable L1 regularization
      interventionPenalty,  // Lambda for L1 penalty
      maxGenerations,
      populationSize,
      sigma: 0.3,
      maxTime: simTime,
      ensembleSize: 8,  // Sample cells at different spatial positions
      targetLineage,  // 0 = all positions, 1-6 = specific lineage
      uniformMorphogens,  // Apply morphogens to all cells equally
      requireHomogeneous,  // Require ALL cells to match target
      timeCourseMode,  // Enable time-course target matching
    },
    knockoutCandidates,
    modifierCandidates: knockoutCandidates,  // Same genes can have modifiers (TFs, ligands, receptors)
  };

  worker.postMessage(startMsg);

  // Update UI
  isRunning = true;
  updateUIRunning(true, maxGenerations);
}

function handleProgress(msg: WorkerProgressMessage): void {
  fitnessHistory.push({
    gen: msg.generation,
    fitness: msg.bestFitness,
    meanFitness: msg.meanFitness,
  });

  // Update stats
  const maxGen = parseInt((document.getElementById('max-generations') as HTMLInputElement).value);
  const genDisplay = document.getElementById('gen-display');
  const fitnessDisplay = document.getElementById('fitness-display');
  const worstDisplay = document.getElementById('worst-display');
  const corrDisplay = document.getElementById('corr-display');
  const interventionStat = document.getElementById('intervention-stat');
  const interventionDisplay = document.getElementById('intervention-display');

  if (genDisplay) genDisplay.textContent = `${msg.generation} / ${maxGen}`;
  if (fitnessDisplay) fitnessDisplay.textContent = msg.bestFitness.toFixed(4);
  if (worstDisplay) worstDisplay.textContent = msg.worstCellFitness.toFixed(4);
  if (corrDisplay) corrDisplay.textContent = msg.correlation.toFixed(3);

  // Get encoding config
  const includeGlobal = (document.getElementById('opt-global') as HTMLInputElement).checked;
  const includeKnockouts = (document.getElementById('opt-knockouts') as HTMLInputElement).checked;
  const includeMorphogens = (document.getElementById('opt-morphogens') as HTMLInputElement).checked;
  const includeModifiers = (document.getElementById('opt-modifiers') as HTMLInputElement).checked;

  // Show intervention count if modifiers are enabled
  if (interventionStat) {
    interventionStat.style.display = includeModifiers ? 'block' : 'none';
  }
  if (interventionDisplay && includeModifiers) {
    interventionDisplay.textContent = String(msg.interventionCount);
  }

  const config: EncodingConfig = {
    includeGlobal,
    includeKnockouts,
    includeMorphogens,
    includeModifiers,
    geneNames,
    knockoutGeneIndices: knockoutCandidates,
    modifierGeneIndices: knockoutCandidates,
  };

  // Decode and simulate to show expression
  const params = decode(new Float64Array(msg.bestParams), config);
  const simResult = engine.runSimulation(params, Math.PI, parseInt((document.getElementById('sim-time') as HTMLInputElement).value));
  bestSimulatedExpression = simResult.expression;
  bestParams = params;

  updateCharts();
}

function handleDone(msg: WorkerDoneMessage): void {
  bestParams = msg.decodedParams;

  // Run final simulation for display
  const simResult = engine.runSimulation(bestParams, Math.PI, parseInt((document.getElementById('sim-time') as HTMLInputElement).value));
  bestSimulatedExpression = simResult.expression;

  updateResults();
  updateCharts();
  stopOptimization();

  // Update status
  const statusLabel = document.getElementById('status-label');
  if (statusLabel) {
    statusLabel.textContent = msg.result.converged ? 'Converged' : 'Done';
    statusLabel.classList.remove('running');
    statusLabel.classList.add('done');
  }
}

function stopOptimization(): void {
  if (worker) {
    worker.postMessage({ type: 'stop' });
    worker.terminate();
    worker = null;
  }

  isRunning = false;
  updateUIRunning(false, 0);
}

function updateUIRunning(running: boolean, maxGen: number): void {
  const startBtn = document.getElementById('btn-start') as HTMLButtonElement;
  const stopBtn = document.getElementById('btn-stop') as HTMLButtonElement;
  const statusLabel = document.getElementById('status-label');

  startBtn.disabled = running;
  stopBtn.disabled = !running;

  if (running) {
    startBtn.classList.add('running');
    startBtn.textContent = 'Running...';
    if (statusLabel) {
      statusLabel.textContent = 'Running';
      statusLabel.classList.add('running');
      statusLabel.classList.remove('done');
    }
  } else {
    startBtn.classList.remove('running');
    startBtn.textContent = 'Start Optimization';
    if (statusLabel && !statusLabel.classList.contains('done')) {
      statusLabel.textContent = 'Ready';
      statusLabel.classList.remove('running');
    }
  }
}

// Initialize config slider displays
function initializeConfigSliders(): void {
  const sliders = [
    { id: 'max-generations', valId: 'max-gen-val', suffix: '' },
    { id: 'pop-size', valId: 'pop-size-val', suffix: '' },
    { id: 'sim-time', valId: 'sim-time-val', suffix: 'h' },
    { id: 'intervention-penalty', valId: 'intervention-penalty-val', suffix: '' },
  ];

  for (const { id, valId, suffix } of sliders) {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valSpan = document.getElementById(valId);

    if (slider && valSpan) {
      slider.addEventListener('input', () => {
        // Format intervention penalty with 2 decimal places
        if (id === 'intervention-penalty') {
          valSpan.textContent = parseFloat(slider.value).toFixed(2);
        } else {
          valSpan.textContent = slider.value + suffix;
        }
      });
    }
  }

  // Set up modifier sub-options visibility toggle
  const modifierCheckbox = document.getElementById('opt-modifiers') as HTMLInputElement;
  const minimalInterventionGroup = document.getElementById('minimal-intervention-group');
  const interventionPenaltyGroup = document.getElementById('intervention-penalty-group');
  const minimalInterventionCheckbox = document.getElementById('opt-minimal-intervention') as HTMLInputElement;

  function updateModifierSubOptions(): void {
    const modifiersEnabled = modifierCheckbox.checked;
    minimalInterventionGroup?.classList.toggle('visible', modifiersEnabled);

    const minimalEnabled = modifiersEnabled && minimalInterventionCheckbox.checked;
    interventionPenaltyGroup?.classList.toggle('visible', minimalEnabled);
  }

  modifierCheckbox?.addEventListener('change', updateModifierSubOptions);
  minimalInterventionCheckbox?.addEventListener('change', updateModifierSubOptions);

  // Initial state
  updateModifierSubOptions();
}

// Import modal handling
function initializeImportModal(): void {
  const modal = document.getElementById('import-modal');
  const btnImport = document.getElementById('btn-import');
  const btnConfirm = document.getElementById('btn-import-confirm');
  const btnCancel = document.getElementById('btn-import-cancel');
  const closeBtn = modal?.querySelector('.modal-close');
  const textarea = document.getElementById('import-textarea') as HTMLTextAreaElement;

  function showModal() {
    modal?.classList.remove('hidden');
  }

  function hideModal() {
    modal?.classList.add('hidden');
  }

  btnImport?.addEventListener('click', showModal);
  btnCancel?.addEventListener('click', hideModal);
  closeBtn?.addEventListener('click', hideModal);

  btnConfirm?.addEventListener('click', () => {
    try {
      const json = JSON.parse(textarea.value);
      if (json.expression && Array.isArray(json.expression)) {
        targetExpression = new Float32Array(json.expression);
        // Re-initialize sliders with new values
        initializeTargetSliders();
      }
      if (json.weights && Array.isArray(json.weights)) {
        targetWeights = new Float32Array(json.weights);
      }
      hideModal();
      updateCharts();
    } catch (e) {
      alert('Invalid JSON format');
    }
  });
}

// Apply to trajectory button
function initializeApplyButton(): void {
  const btn = document.getElementById('btn-apply');
  btn?.addEventListener('click', () => {
    if (!bestParams) return;

    const encoded = encodeParamsURL(bestParams);
    window.location.href = `/trajectory.html?params=${encoded}`;
  });
}

// Export button
function initializeExportButton(): void {
  const btn = document.getElementById('btn-export');
  btn?.addEventListener('click', () => {
    if (!bestParams) return;

    const json = exportParamsJSON(bestParams);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'inferred-params.json';
    a.click();

    URL.revokeObjectURL(url);
  });
}

// Initialize sensitivity analysis button
function initializeSensitivityButton(): void {
  const btn = document.getElementById('btn-sensitivity');
  btn?.addEventListener('click', runSensitivityAnalysis);
}

// Main initialization
function init(): void {
  initializeDefaultTarget();
  initializeTargetSliders();
  initializeStatePresets();
  initializeTimeCourseUI();
  initializeTimepointModal();
  initializeGeneSelectionUI();
  initializeWeightControls();
  initializeAccordions();
  initializeCanvases();
  initializeConfigSliders();
  initializeImportModal();
  initializeApplyButton();
  initializeExportButton();
  initializeSensitivityButton();
  initializeStabilityButton();

  // Initialize weights based on default preset
  updateWeights();

  // Button handlers
  document.getElementById('btn-start')?.addEventListener('click', startOptimization);
  document.getElementById('btn-stop')?.addEventListener('click', stopOptimization);

  updateCharts();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
