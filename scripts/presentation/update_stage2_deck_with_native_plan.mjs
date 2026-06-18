import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const artifactToolPath = require.resolve("@oai/artifact-tool", {
  paths: [
    process.env.NODE_PATH || "",
    path.join(process.env.USERPROFILE || os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "node", "node_modules"),
  ].filter(Boolean),
});
const { PresentationFile, FileBlob } = await import(pathToFileURL(artifactToolPath).href);

const ROOT = process.cwd();
const SOURCE_PPTX = process.env.SOURCE_PPTX || "C:/Users/13917/Desktop/ascr_stage2_progress_advisor_meeting.pptx";
const OUT_DIR = path.join(ROOT, "outputs", "presentations");
const PREVIEW_DIR = path.join(OUT_DIR, "advisor_native_update_preview");
const FINAL_PPTX = path.join(OUT_DIR, "ascr_stage2_progress_advisor_meeting_lumina_native_next.pptx");

const W = 1280;
const H = 720;
const C = {
  bg: "#111827",
  panel: "#1F2937",
  panel2: "#0F172A",
  cyan: "#22D3EE",
  amber: "#F59E0B",
  green: "#22C55E",
  red: "#F87171",
  white: "#F9FAFB",
  text: "#E5E7EB",
  muted: "#9CA3AF",
  line: "#374151",
};

function textOf(shape) {
  try {
    return shape.toSnapshot()?.text || "";
  } catch {
    return "";
  }
}

function replaceText(deck, replacements) {
  for (const slide of deck.slides.items) {
    for (const shape of slide.shapes.items || []) {
      const text = textOf(shape);
      if (Object.prototype.hasOwnProperty.call(replacements, text)) {
        shape.text = replacements[text];
      }
    }
  }
}

function renumberExistingSlides(deck) {
  for (const [index, slide] of deck.slides.items.entries()) {
    for (const shape of slide.shapes.items || []) {
      const snap = shape.toSnapshot?.();
      const text = snap?.text || "";
      const frame = snap?.frame || {};
      if (/^\d{2}$/.test(text) && frame.left > 1080 && frame.top > 630) {
        shape.text = String(index + 1).padStart(2, "0");
      }
    }
  }
}

function addText(slide, text, { x, y, w, h, size = 22, color = C.text, bold = false, align = "left" }) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: size,
    color,
    bold,
    typeface: bold ? "Aptos Display" : "Aptos",
    alignment: align,
    fit: "shrink",
  };
  return shape;
}

function addBox(slide, x, y, w, h, line = C.line, fill = C.panel2) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
  });
}

function addArrow(slide, x, y, w = 64, h = 34, fill = C.cyan) {
  slide.shapes.add({
    geometry: "rightArrow",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function baseSlide(deck) {
  const slide = deck.slides.add();
  slide.background.fill = C.bg;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: W, height: 8 },
    fill: C.cyan,
    line: { style: "solid", fill: C.cyan, width: 0 },
  });
  return slide;
}

function addTitle(slide, title, subtitle) {
  addText(slide, "ASCR Stage-2 Progress", { x: 64, y: 34, w: 360, h: 24, size: 13, color: C.cyan, bold: true });
  addText(slide, title, { x: 64, y: 80, w: 970, h: 66, size: 38, color: C.white, bold: true });
  if (subtitle) addText(slide, subtitle, { x: 66, y: 150, w: 940, h: 36, size: 18, color: C.muted });
}

function footer(slide, n) {
  addText(slide, "Sources: ASCR repo docs, AI_COLLAB_LOG, current implementation", { x: 64, y: 682, w: 760, h: 20, size: 10, color: C.muted });
  addText(slide, String(n).padStart(2, "0"), { x: 1160, y: 674, w: 56, h: 28, size: 14, color: C.cyan, bold: true, align: "right" });
}

function node(slide, label, sub, x, y, w, color = C.cyan) {
  addBox(slide, x, y, w, 96, color);
  addText(slide, label, { x: x + 16, y: y + 16, w: w - 32, h: 28, size: 18, color: C.white, bold: true, align: "center" });
  addText(slide, sub, { x: x + 16, y: y + 50, w: w - 32, h: 32, size: 12, color: C.muted, align: "center" });
}

function bullets(slide, items, x, y, w, lineH = 44, size = 20) {
  items.forEach((item, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      position: { left: x, top: y + i * lineH + 8, width: 8, height: 8 },
      fill: C.cyan,
      line: { style: "solid", fill: C.cyan, width: 0 },
    });
    addText(slide, item, { x: x + 24, y: y + i * lineH, w, h: lineH, size, color: C.text });
  });
}

function metric(slide, label, value, sub, x, y, accent = C.cyan) {
  addBox(slide, x, y, 310, 126, "#2D3748", C.panel);
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: 5, height: 126 },
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
  });
  addText(slide, label, { x: x + 18, y: y + 16, w: 270, h: 24, size: 13, color: C.muted, bold: true });
  addText(slide, value, { x: x + 18, y: y + 42, w: 270, h: 44, size: 32, color: C.white, bold: true });
  addText(slide, sub, { x: x + 18, y: y + 90, w: 270, h: 28, size: 13, color: C.text });
}

function note(slide, n, title, body) {
  slide.speakerNotes.text = `Slide ${n}. ${title}\n\n${body}`;
}

function appendNativePlan(deck, startNumber) {
  {
    const s = baseSlide(deck);
    addTitle(s, "Correction: formal Stage 2 target", "The student should be Lumina-native, not an external shallow localizer.");
    metric(s, "Old scaffold", "external", "RGB/prompt-hash localizer", 92, 250, C.amber);
    metric(s, "Formal student", "native", "Lumina reads image + prompt", 486, 250, C.green);
    metric(s, "Selector", "fixed", "cells -> reopen mask", 880, 250, C.cyan);
    bullets(s, [
      "Keep grid-localizer-v0/v1 as historical scaffold baselines only.",
      "Distill Qwen3.7-style SemanticEvaluation JSON into Lumina-DiMOO itself.",
      "Do not benchmark external scaffold as the formal after-distill model."
    ], 112, 470, 1040, 44, 20);
    footer(s, startNumber);
    note(s, startNumber, "Formal Stage 2 target", "这里要明确纠正：之前的 external localizer 只是 scaffold。正式 Stage 2 是让 Lumina-DiMOO 自己读取 image 和 prompt，输出 SemanticEvaluation JSON。");
  }
  {
    const s = baseSlide(deck);
    addTitle(s, "Target ASCR loop", "A single Lumina path generates, evaluates, and reopens.");
    node(s, "Prompt", "text condition", 64, 292, 136, C.cyan);
    addArrow(s, 216, 323, 46, 34);
    node(s, "Lumina generate", "initial image/tokens", 278, 292, 196, C.green);
    addArrow(s, 490, 323, 46, 34);
    node(s, "Lumina evaluate", "JSON: has_error + cells", 552, 292, 224, C.amber);
    addArrow(s, 792, 323, 46, 34);
    node(s, "Selector", "cells -> token mask", 854, 292, 166, C.cyan);
    addArrow(s, 1036, 323, 46, 34);
    node(s, "Reopen", "continue Lumina", 1098, 292, 160, C.green);
    addBox(s, 330, 500, 620, 78, C.line, C.panel);
    addText(s, "Stop: no error / malformed JSON abstention / no actionable cells / max_iterations", { x: 360, y: 522, w: 560, h: 28, size: 18, color: C.white, bold: true, align: "center" });
    footer(s, startNumber + 1);
    note(s, startNumber + 1, "Target ASCR loop", "目标流程是 Lumina generate，然后 Lumina-native evaluator 输出 JSON，selector 把 cells 映射成 token mask，最后 Lumina reopen。");
  }
  {
    const s = baseSlide(deck);
    addTitle(s, "What has been implemented now", "The repo is ready for a feasibility gate, not fake LoRA claims.");
    bullets(s, [
      "Added lumina_native_evaluator backend: parse JSON if a native answer hook exists; otherwise abstain safely.",
      "Added lumina_native_audit CLI to check whether Lumina exposes image-conditioned text/MMU answering.",
      "Added SFT data preparation from Qwen3.7 teacher labels for future Lumina LoRA/SFT.",
      "Updated docs/handoff so server AI stops if the native hook is missing."
    ], 112, 240, 1040, 58, 20);
    addBox(s, 160, 560, 960, 56, C.amber, C.panel);
    addText(s, "No formal Stage-2 benchmark should run until the audit confirms a native evaluator hook.", { x: 190, y: 576, w: 900, h: 28, size: 19, color: C.white, bold: true, align: "center" });
    footer(s, startNumber + 2);
    note(s, startNumber + 2, "Implemented now", "这次代码层面加的是 feasibility gate 和数据准备，不是假装已经完成 LoRA。服务器要先证明 Lumina 有 image-conditioned text output hook。");
  }
  {
    const s = baseSlide(deck);
    addTitle(s, "Server AI next steps", "Run the gate, prepare SFT smoke data, then report the blocker or proceed.");
    bullets(s, [
      "1. Pull latest main and activate .venv-lumina.",
      "2. Run scripts/training/run_lumina_native_audit.sh.",
      "3. Run scripts/training/prepare_lumina_native_sft.sh with LIMIT=10.",
      "4. Append audit JSON and SFT manifest summary to AI_COLLAB_LOG.",
      "5. If no native answer hook exists, stop before formal Stage-2 benchmarks."
    ], 120, 235, 1000, 54, 21);
    footer(s, startNumber + 3);
    note(s, startNumber + 3, "Server AI next steps", "服务器 AI 的任务是先跑 audit，再准备十条 SFT smoke 数据，然后把结果写回 AI_COLLAB_LOG。如果没有 native hook，就停下来报告 blocker。");
  }
  {
    const s = baseSlide(deck);
    addTitle(s, "Benchmark policy after the gate", "Only compare direct Lumina vs Lumina-native ASCR.");
    addBox(s, 116, 250, 440, 230, C.cyan);
    addText(s, "Before", { x: 148, y: 286, w: 260, h: 34, size: 28, color: C.cyan, bold: true });
    addText(s, "Lumina direct generation", { x: 148, y: 358, w: 330, h: 44, size: 28, color: C.white, bold: true });
    addBox(s, 724, 250, 440, 230, C.green);
    addText(s, "After", { x: 756, y: 286, w: 260, h: 34, size: 28, color: C.green, bold: true });
    addText(s, "Lumina-native evaluator + selector + ASCR loop", { x: 756, y: 344, w: 330, h: 78, size: 25, color: C.white, bold: true });
    addText(s, "External grid-localizer scaffold is not a formal benchmark arm.", { x: 210, y: 548, w: 860, h: 34, size: 24, color: C.amber, bold: true, align: "center" });
    footer(s, startNumber + 4);
    note(s, startNumber + 4, "Benchmark policy", "正式 benchmark 只比较 before 和 after：before 是 Lumina direct generation；after 是 Lumina-native evaluator 加 selector 加 ASCR loop。");
  }
}

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });

const deck = await PresentationFile.importPptx(await FileBlob.load(SOURCE_PPTX));

replaceText(deck, {
  "From selective semantic reopening to distilled student localizers": "From selective semantic reopening to Lumina-native semantic evaluation",
  "student semantic localizer": "Lumina-native evaluator",
  "Student localizer v0": "External scaffold v0",
  "First end-to-end student baseline: meaningful, but intentionally lightweight.": "First end-to-end scaffold baseline: useful, but not the formal student.",
  "v0 student": "v0 scaffold",
  "Purpose: validate the Stage-2 measurement and deployment path.": "Purpose: validate the measurement path, not claim final distillation.",
  "v0 localizer": "scaffold v0",
  "End-to-end Stage-2 scaffold is running: train student → generate before/after → judge with Qwen3.7.": "End-to-end scaffold is running: train external localizer → generate before/after → judge with Qwen3.7.",
  "Current v0 is a weak but valid baseline; it verifies the scientific measurement loop.": "Current v0 is a weak scaffold baseline; it verifies the measurement loop only.",
  "Next: add teacher labels on benchmark images and train grid-localizer-v1.": "Next: pivot to Lumina-native evaluator distillation.",
  "Student localizer v1: data and model upgrade": "Historical scaffold v1: data/model upgrade",
  "Use benchmark images as additional teacher-labeled training data.": "Retained as scaffold/debugging, not the formal Stage-2 student.",
  "grid-localizer-v1": "scaffold-v1",
  "Purpose: decide whether shallow student modeling is enough before moving to neural Stage 2.": "Purpose: debug data flow before Lumina-native evaluator distillation."
});

const existingSlides = deck.slides.items.length;
renumberExistingSlides(deck);
appendNativePlan(deck, existingSlides + 1);

await fs.mkdir(PREVIEW_DIR, { recursive: true });
for (const [index, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(PREVIEW_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
}
const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(PREVIEW_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(FINAL_PPTX);

console.log(JSON.stringify({
  source: SOURCE_PPTX,
  output: FINAL_PPTX,
  slides: deck.slides.items.length,
  preview: path.join(PREVIEW_DIR, "deck-montage.webp"),
}, null, 2));
