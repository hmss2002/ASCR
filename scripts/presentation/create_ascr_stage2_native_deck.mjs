import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const artifactToolPath = require.resolve("@oai/artifact-tool", {
  paths: [
    process.env.NODE_PATH || "",
    path.join(process.env.USERPROFILE || "", ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "node", "node_modules"),
  ].filter(Boolean),
});
const { Presentation, PresentationFile } = await import(pathToFileURL(artifactToolPath).href);

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "outputs", "presentations");
const PREVIEW_DIR = path.join(OUT_DIR, "native_preview");
const FINAL_PPTX = path.join(OUT_DIR, "ascr_stage2_lumina_native_distillation_plan.pptx");
const NOTES_TXT = path.join(OUT_DIR, "ascr_stage2_lumina_native_distillation_notes.txt");

const W = 1280;
const H = 720;
const C = {
  bg: "#0F172A",
  panel: "#1E293B",
  panel2: "#111827",
  cyan: "#22D3EE",
  green: "#22C55E",
  amber: "#F59E0B",
  red: "#F87171",
  white: "#F8FAFC",
  text: "#E5E7EB",
  muted: "#94A3B8",
  line: "#334155",
};

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

function addArrow(slide, x, y, w = 70, h = 34, fill = C.cyan) {
  slide.shapes.add({
    geometry: "rightArrow",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, "ASCR Stage 2", { x: 64, y: 34, w: 260, h: 24, size: 13, color: C.cyan, bold: true });
  addText(slide, title, { x: 64, y: 78, w: 960, h: 72, size: 38, color: C.white, bold: true });
  if (subtitle) addText(slide, subtitle, { x: 66, y: 148, w: 900, h: 38, size: 18, color: C.muted });
}

function footer(slide, n, source = "Sources: ASCR repo docs, AI_COLLAB_LOG, current implementation") {
  addText(slide, source, { x: 64, y: 682, w: 780, h: 20, size: 10, color: C.muted });
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

const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const notes = [];

function slideBase() {
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: W, height: 8 },
    fill: C.cyan,
    line: { style: "solid", fill: C.cyan, width: 0 },
  });
  return slide;
}

function addNote(slide, n, title, body) {
  const text = `Slide ${n}. ${title}\n\n${body}`;
  slide.speakerNotes.text = text;
  notes.push(text + "\n");
}

{
  const s = slideBase();
  addText(s, "ASCR", { x: 72, y: 62, w: 180, h: 32, size: 20, color: C.cyan, bold: true });
  addText(s, "Stage 2 Pivot", { x: 72, y: 130, w: 780, h: 70, size: 56, color: C.white, bold: true });
  addText(s, "Distill Lumina-DiMOO into its own semantic evaluator, not an external shallow localizer.", { x: 76, y: 214, w: 930, h: 38, size: 23, color: C.text });
  node(s, "Teacher", "Qwen3.7 JSON labels", 90, 385, 250, C.amber);
  addArrow(s, 365, 416, 78, 36);
  node(s, "Student", "Lumina-native evaluator", 470, 385, 300, C.green);
  addArrow(s, 795, 416, 78, 36);
  node(s, "Loop", "selector + reopen", 900, 385, 250, C.cyan);
  footer(s, 1, "Project: Alternating Semantic-Confidence Revision");
  addNote(s, 1, "Title", "老师，这次我把 Stage 2 的定义纠正过来。真正要蒸馏的不是外置的 shallow localizer，而是 Lumina-DiMOO 自己的 semantic evaluator / MMU-like ability。");
}

{
  const s = slideBase();
  addTitle(s, "Correction: what the student should be", "The old localizer was useful infrastructure, but not the final research target.");
  metric(s, "Old scaffold", "external", "RGB/prompt-hash localizer", 92, 246, C.amber);
  metric(s, "Formal student", "native", "Lumina reads image + prompt", 486, 246, C.green);
  metric(s, "Selector", "fixed", "cells -> reopen mask", 880, 246, C.cyan);
  bullets(s, [
    "The distilled capability should live in the Lumina model path.",
    "Output remains strict SemanticEvaluation JSON.",
    "GridSemanticReopeningSelector stays fixed for now."
  ], 116, 458, 1020, 46, 20);
  footer(s, 2);
  addNote(s, 2, "Student definition", "这里要讲清楚：v0/v1 localizer 是 scaffold，不是最终 student。正式 student 应该是 Lumina-native evaluator，也就是 Lumina 自己看图和 prompt，然后输出 JSON。");
}

{
  const s = slideBase();
  addTitle(s, "Target ASCR loop", "A single Lumina path generates, evaluates, and reopens.");
  node(s, "Prompt", "text condition", 64, 294, 136, C.cyan);
  addArrow(s, 216, 324, 46, 34);
  node(s, "Lumina generate", "initial image/tokens", 278, 294, 196, C.green);
  addArrow(s, 490, 324, 46, 34);
  node(s, "Lumina evaluate", "JSON: has_error + cells", 552, 294, 224, C.amber);
  addArrow(s, 792, 324, 46, 34);
  node(s, "Selector", "cells -> token mask", 854, 294, 166, C.cyan);
  addArrow(s, 1036, 324, 46, 34);
  node(s, "Reopen", "continue Lumina", 1098, 294, 160, C.green);
  addBox(s, 330, 500, 620, 78, C.line, C.panel);
  addText(s, "Stop: no error / malformed JSON abstention / no actionable cells / max_iterations", { x: 360, y: 522, w: 560, h: 28, size: 18, color: C.white, bold: true, align: "center" });
  footer(s, 3);
  addNote(s, 3, "Target loop", "目标流程是 Lumina generate，然后 Lumina-native evaluator 输出 JSON，selector 把 cells 映射成 token mask，再由 Lumina reopen。这里不再把外置 shallow localizer 当主角。");
}

{
  const s = slideBase();
  addTitle(s, "Feasibility gate", "Before training, prove Lumina can answer from image/image tokens.");
  addBox(s, 100, 235, 480, 300, C.amber);
  addText(s, "Required hook", { x: 132, y: 270, w: 360, h: 30, size: 25, color: C.amber, bold: true });
  bullets(s, [
    "answer_image(question, image)",
    "or answer_vq_tokens(question, codes)",
    "or equivalent MMU text output"
  ], 132, 340, 360, 42, 18);
  addBox(s, 700, 235, 480, 300, C.green);
  addText(s, "Safe default", { x: 732, y: 270, w: 360, h: 30, size: 25, color: C.green, bold: true });
  bullets(s, [
    "If hook is missing: abstain",
    "Malformed JSON: abstain",
    "No unsafe reopen from guesses"
  ], 732, 340, 360, 42, 18);
  footer(s, 4);
  addNote(s, 4, "Feasibility gate", "第一步不是直接训练，而是确认 Lumina-DiMOO wrapper 是否真的支持 image-conditioned text generation。如果没有这个 hook，就必须停下来，不能假装 RGB localizer 是正式 student。");
}

{
  const s = slideBase();
  addTitle(s, "Teacher supervision", "Qwen3.7-plus produces the JSON behavior Lumina should imitate.");
  node(s, "prompt + image", "current ASCR state", 100, 320, 220, C.cyan);
  addArrow(s, 350, 350, 70, 34);
  node(s, "Qwen3.7 teacher", "offline/API labels", 450, 320, 250, C.amber);
  addArrow(s, 730, 350, 70, 34);
  node(s, "target JSON", "SemanticEvaluation", 830, 320, 260, C.green);
  bullets(s, [
    "Fields: has_error, regions.cells, reason, confidence, correction_instruction.",
    "Teacher labels are collected offline on the login node.",
    "No API key enters compute-node generation or final ASCR inference."
  ], 126, 508, 1000, 38, 18);
  footer(s, 5);
  addNote(s, 5, "Teacher supervision", "Teacher 还是 Qwen3.7-plus。它负责产生 prompt-image 到 SemanticEvaluation JSON 的监督信号。之后 Lumina 要模仿这个输出格式和判断能力。");
}

{
  const s = slideBase();
  addTitle(s, "Training path", "Prepare SFT rows now; LoRA starts only after the native hook is verified.");
  node(s, "teacher dataset", "localization labels", 88, 330, 210, C.amber);
  addArrow(s, 320, 360, 64, 34);
  node(s, "SFT examples", "input_text + image + target_json", 408, 330, 270, C.cyan);
  addArrow(s, 700, 360, 64, 34);
  node(s, "LoRA/SFT smoke", "blocked until hook exists", 788, 330, 270, C.red);
  addBox(s, 138, 520, 1000, 60, C.line, C.panel);
  addText(s, "Current implementation prepares data and records the blocker instead of pretending training is complete.", { x: 170, y: 538, w: 940, h: 28, size: 20, color: C.white, bold: true, align: "center" });
  footer(s, 6);
  addNote(s, 6, "Training path", "现在代码先准备 SFT examples：输入是 evaluator prompt 和 image，target 是 Qwen 的 JSON。真正 LoRA/SFT 要等 Lumina native answer hook 被验证后再开始。");
}

{
  const s = slideBase();
  addTitle(s, "Benchmark policy", "Only compare direct Lumina vs Lumina-native ASCR.");
  addBox(s, 116, 250, 440, 230, C.cyan);
  addText(s, "Before", { x: 148, y: 286, w: 260, h: 34, size: 28, color: C.cyan, bold: true });
  addText(s, "Lumina direct generation", { x: 148, y: 358, w: 330, h: 44, size: 28, color: C.white, bold: true });
  addBox(s, 724, 250, 440, 230, C.green);
  addText(s, "After", { x: 756, y: 286, w: 260, h: 34, size: 28, color: C.green, bold: true });
  addText(s, "Lumina-native evaluator + selector + ASCR loop", { x: 756, y: 344, w: 330, h: 78, size: 25, color: C.white, bold: true });
  addText(s, "External grid-localizer scaffold is not a formal benchmark arm.", { x: 210, y: 548, w: 860, h: 34, size: 24, color: C.amber, bold: true, align: "center" });
  footer(s, 7);
  addNote(s, 7, "Benchmark policy", "正式 benchmark 只比较 before 和 after：before 是 Lumina direct generation；after 是 Lumina-native evaluator 加 selector 加 ASCR loop。外置 localizer 不作为正式 benchmark arm。");
}

{
  const s = slideBase();
  addTitle(s, "Server execution plan", "What the server AI should run next.");
  bullets(s, [
    "1. Pull main and activate .venv-lumina.",
    "2. Run scripts/training/run_lumina_native_audit.sh.",
    "3. Prepare SFT smoke data with scripts/training/prepare_lumina_native_sft.sh.",
    "4. Append audit JSON and SFT manifest to AI_COLLAB_LOG.",
    "5. If native hook is missing, stop and report the blocker."
  ], 120, 245, 1000, 54, 21);
  footer(s, 8);
  addNote(s, 8, "Server handoff", "服务器下一步不是跑 grid-localizer-v1 benchmark，而是先做 Lumina-native audit。如果没有 native answer hook，就记录 blocker，先不要做正式 Stage 2 benchmark。");
}

{
  const s = slideBase();
  addTitle(s, "Decision criteria", "What evidence decides the next engineering branch.");
  metric(s, "Audit", "hook?", "image-conditioned text output", 90, 250, C.amber);
  metric(s, "Smoke", "JSON", "valid SemanticEvaluation", 486, 250, C.cyan);
  metric(s, "Benchmark", "delta", "before vs native-ASCR", 882, 250, C.green);
  bullets(s, [
    "If hook exists: implement LoRA/SFT smoke and run native before/after.",
    "If hook is absent: add or expose MMU answer path before further Stage-2 claims.",
    "Do not present shallow external localizer as the final student."
  ], 112, 470, 1040, 44, 20);
  footer(s, 9);
  addNote(s, 9, "Decision criteria", "判断标准很清楚：先看 hook 是否存在，再看 JSON 输出是否合法，最后再看 before/after benchmark 是否有提升。如果 hook 不存在，下一步是补能力，不是继续包装 shallow localizer。");
}

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });

await fs.writeFile(
  NOTES_TXT,
  "\uFEFF" + [
    "ASCR Stage-2 Lumina-Native Distillation - Speaker Notes",
    "Style: 中文讲解为主，关键术语保留 English。",
    "",
    ...notes,
  ].join("\n"),
  "utf8",
);

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(PREVIEW_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
}

const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(PREVIEW_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL_PPTX);

console.log(JSON.stringify({
  pptx: FINAL_PPTX,
  notes: NOTES_TXT,
  slides: presentation.slides.items.length,
  preview: path.join(PREVIEW_DIR, "deck-montage.webp"),
}, null, 2));
