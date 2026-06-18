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
const FINAL_PPTX = path.join(OUT_DIR, "ascr_stage2_progress_advisor_meeting.pptx");
const NOTES_TXT = path.join(OUT_DIR, "ascr_stage2_progress_advisor_speaker_notes.txt");
const QA_DIR = path.join(OUT_DIR, "preview");

const W = 1280;
const H = 720;
const C = {
  bg: "#111827",
  panel: "#1F2937",
  panel2: "#0F172A",
  panel3: "#172033",
  cyan: "#22D3EE",
  cyan2: "#0891B2",
  amber: "#F59E0B",
  green: "#22C55E",
  red: "#EF4444",
  white: "#F9FAFB",
  text: "#E5E7EB",
  muted: "#9CA3AF",
  line: "#374151",
};

const metrics = JSON.parse(await fs.readFile(path.join(ROOT, "outputs", "stage2_students", "grid_localizer_v0", "metrics.json"), "utf8"));
const judgeIn = JSON.parse(await fs.readFile(path.join(ROOT, "outputs", "api_judges", "student_localizer_v0", "in_domain_hard64_holdout", "summary.json"), "utf8"));
const judgeGe = JSON.parse(await fs.readFile(path.join(ROOT, "outputs", "api_judges", "student_localizer_v0", "geneval_smoke16", "summary.json"), "utf8"));

function pct(value, digits = 1) {
  if (value === null || value === undefined) return "n/a";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function n(value, digits = 3) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toFixed(digits);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addBox(slide, { x, y, w, h, fill = C.panel, line = C.line, radius = "rounded-lg", name }) {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addText(slide, text, { x, y, w, h, size = 22, color = C.text, bold = false, align = "left", name }) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    fontSize: size,
    color,
    bold,
    typeface: bold ? "Aptos Display" : "Aptos",
    alignment: align,
    fit: "shrink",
  };
  return s;
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, "ASCR Stage-2 Progress", { x: 64, y: 34, w: 420, h: 24, size: 13, color: C.cyan, bold: true });
  addText(slide, title, { x: 64, y: 74, w: 860, h: 78, size: 38, color: C.white, bold: true });
  if (subtitle) addText(slide, subtitle, { x: 66, y: 144, w: 820, h: 38, size: 18, color: C.muted });
}

function addFooter(slide, num, source = "Sources: ASCR repo README, AI_COLLAB_LOG, result JSON summaries") {
  addText(slide, source, { x: 64, y: 680, w: 760, h: 20, size: 10, color: C.muted });
  addText(slide, String(num).padStart(2, "0"), { x: 1160, y: 674, w: 56, h: 28, size: 14, color: C.cyan, bold: true, align: "right" });
}

function addMetric(slide, label, value, sub, { x, y, w = 250, h = 126, accent = C.cyan }) {
  addBox(slide, { x, y, w, h, fill: C.panel, line: "#2D3748" });
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: 5, height: h },
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
  });
  addText(slide, label, { x: x + 18, y: y + 16, w: w - 34, h: 24, size: 13, color: C.muted, bold: true });
  addText(slide, value, { x: x + 18, y: y + 43, w: w - 34, h: 44, size: 34, color: C.white, bold: true });
  addText(slide, sub, { x: x + 18, y: y + 90, w: w - 34, h: 28, size: 13, color: C.text });
}

function addArrow(slide, x, y, w, h, fill = C.cyan) {
  slide.shapes.add({
    geometry: "rightArrow",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addPipelineNode(slide, label, sub, x, y, w, accent = C.cyan) {
  addBox(slide, { x, y, w, h: 98, fill: C.panel2, line: accent });
  addText(slide, label, { x: x + 16, y: y + 16, w: w - 32, h: 28, size: 18, color: C.white, bold: true, align: "center" });
  addText(slide, sub, { x: x + 16, y: y + 50, w: w - 32, h: 34, size: 12, color: C.muted, align: "center" });
}

function addBullets(slide, items, x, y, w, lineH = 36, size = 20) {
  items.forEach((item, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      position: { left: x, top: y + i * lineH + 8, width: 8, height: 8 },
      fill: C.cyan,
      line: { style: "solid", fill: C.cyan, width: 0 },
    });
    addText(slide, item, { x: x + 22, y: y + i * lineH, w, h: lineH, size, color: C.text });
  });
}

function addMiniBar(slide, label, value, max, x, y, w, color) {
  addText(slide, label, { x, y: y - 2, w: 180, h: 24, size: 14, color: C.text });
  addBox(slide, { x: x + 190, y, w, h: 16, fill: "#0B1220", line: "#243044", radius: "rounded-sm" });
  slide.shapes.add({
    geometry: "rect",
    position: { left: x + 190, top: y, width: Math.max(2, w * value / max), height: 16 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
  addText(slide, String(value), { x: x + 198 + w, y: y - 3, w: 40, h: 22, size: 13, color: C.muted });
}

const notes = [];
function note(slideNo, title, body) {
  notes.push(`Slide ${slideNo}. ${title}\n${body}\n`);
}

const p = Presentation.create({ slideSize: { width: W, height: H } });

function baseSlide() {
  const slide = p.slides.add();
  slide.background.fill = C.bg;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: W, height: 8 },
    fill: C.cyan,
    line: { style: "solid", fill: C.cyan, width: 0 },
  });
  return slide;
}

// 1
{
  const s = baseSlide();
  slideGlow(s);
  addText(s, "ASCR", { x: 70, y: 60, w: 200, h: 40, size: 20, color: C.cyan, bold: true });
  addText(s, "Stage-2 Progress", { x: 70, y: 134, w: 790, h: 70, size: 54, color: C.white, bold: true });
  addText(s, "From selective semantic reopening to distilled student localizers", { x: 74, y: 214, w: 820, h: 36, size: 24, color: C.text });
  addPipelineNode(s, "Stage 1", "zero-training ASCR", 78, 380, 230, C.cyan);
  addArrow(s, 330, 408, 74, 38, C.cyan2);
  addPipelineNode(s, "Teacher", "Qwen3.7 localization", 430, 380, 230, C.amber);
  addArrow(s, 682, 408, 74, 38, C.cyan2);
  addPipelineNode(s, "Stage 2", "student semantic localizer", 782, 380, 310, C.green);
  addText(s, "Advisor meeting update", { x: 74, y: 626, w: 360, h: 28, size: 16, color: C.muted });
  addFooter(s, 1, "Project: Alternating Semantic-Confidence Revision");
  note(1, "Title", "老师，这次主要汇报 ASCR 从 Stage 1 往 Stage 2 走的进展。核心不是再做一个 heuristic selector，而是把 external semantic evaluator 蒸馏成本地 student semantic localizer。");
}

// 2
{
  const s = baseSlide();
  addTitle(s, "Executive summary", "What has been built, what the first student shows, and why v1 is the right next step.");
  addMetric(s, "Stage-1 Lumina ASCR", "+1.6pp", "82.8% → 84.4% clean", { x: 74, y: 250, accent: C.green });
  addMetric(s, "Teacher dataset", "64 + 79", "quality + localization labels", { x: 370, y: 250, accent: C.amber });
  addMetric(s, "v0 localizer", "0.20", "holdout hit_any rate", { x: 666, y: 250, accent: C.cyan });
  addMetric(s, "Image judge", "+0.012", "in-domain mean delta", { x: 962, y: 250, accent: C.green });
  addBullets(s, [
    "End-to-end Stage-2 scaffold is running: train student → generate before/after → judge with Qwen3.7.",
    "Current v0 is a weak but valid baseline; it verifies the scientific measurement loop.",
    "Next: add teacher labels on benchmark images and train grid-localizer-v1."
  ], 90, 450, 1060, 46, 19);
  addFooter(s, 2);
  note(2, "Executive summary", "一句话总结：Stage 1 的 ASCR loop 已经跑通，Qwen3.7 teacher 数据也有了。现在第一个 end-to-end student localizer baseline 已经在 server 上跑过，但结果还弱，所以不是重复跑 v0，而是进入 v1 data expansion 和 stronger lightweight student。");
}

// 3
{
  const s = baseSlide();
  addTitle(s, "Research problem: confidence ≠ semantic correctness", "A region may be confidence-stable while still violating the prompt.");
  addBox(s, { x: 92, y: 240, w: 480, h: 230, fill: C.panel2, line: C.cyan });
  addText(s, "Native confidence asks", { x: 120, y: 270, w: 390, h: 32, size: 24, color: C.cyan, bold: true });
  addText(s, "Where is the model uncertain?", { x: 120, y: 332, w: 380, h: 54, size: 32, color: C.white, bold: true });
  addText(s, "Good for denoising dynamics, but not equivalent to prompt correctness.", { x: 120, y: 404, w: 380, h: 42, size: 17, color: C.muted });
  addBox(s, { x: 708, y: 240, w: 480, h: 230, fill: C.panel2, line: C.amber });
  addText(s, "ASCR asks", { x: 736, y: 270, w: 390, h: 32, size: 24, color: C.amber, bold: true });
  addText(s, "Where is the image semantically wrong?", { x: 736, y: 322, w: 395, h: 78, size: 30, color: C.white, bold: true });
  addText(s, "Targets count, relation, color binding, missing/extra object errors.", { x: 736, y: 416, w: 390, h: 42, size: 17, color: C.muted });
  addArrow(s, 595, 326, 88, 42, C.cyan2);
  addFooter(s, 3);
  note(3, "Research problem", "原始 masked/discrete generation 主要依赖 confidence。问题是，有些区域 confidence 已经 stable，但 semantic 上仍然错，比如颜色、数量、左右关系。这就是 ASCR 要解决的 confidence-semantic inconsistency。");
}

// 4
{
  const s = baseSlide();
  addTitle(s, "ASCR core loop", "Alternating native confidence refinement with semantic reopening.");
  const y = 285;
  addPipelineNode(s, "Prompt", "text condition", 64, y, 140, C.cyan);
  addArrow(s, 218, y + 30, 48, 34);
  addPipelineNode(s, "Generator", "decode tokens", 278, y, 170, C.green);
  addArrow(s, 462, y + 30, 48, 34);
  addPipelineNode(s, "Semantic block", "localize errors", 522, y, 210, C.amber);
  addArrow(s, 746, y + 30, 48, 34);
  addPipelineNode(s, "Selector", "cells -> reopen mask", 806, y, 170, C.cyan);
  addArrow(s, 990, y + 30, 48, 34);
  addPipelineNode(s, "Reopen", "continue generation", 1050, y, 170, C.green);
  addBox(s, { x: 342, y: 480, w: 600, h: 86, fill: C.panel3, line: C.line });
  addText(s, "Stop conditions: no semantic error / no actionable region / max_iterations", { x: 370, y: 506, w: 548, h: 32, size: 20, color: C.white, bold: true, align: "center" });
  addFooter(s, 4);
  note(4, "ASCR core loop", "ASCR 的核心是 alternating。Confidence block 继续 native denoising；Semantic block 问另一个问题：当前图像哪里和 prompt 不一致，然后只 reopen those selected token regions。");
}

// 5
{
  const s = baseSlide();
  addTitle(s, "Research roadmap", "Stage 2 is about learning the semantic localizer/evaluator.");
  const cards = [
    ["Stage 1", "Zero-training ASCR", "External/self semantic evaluator; grid cells projected to token reopen masks.", C.cyan],
    ["Stage 2", "Distilled student localizer", "Train a local model to predict semantic error cells from prompt + image.", C.amber],
    ["Stage 3", "Cross-model ASCR", "Generalize across stronger masked/discrete generators and evaluators.", C.green],
  ];
  cards.forEach(([k, t, b, color], i) => {
    const x = 96 + i * 386;
    addBox(s, { x, y: 245, w: 320, h: 285, fill: C.panel2, line: color });
    addText(s, k, { x: x + 24, y: 276, w: 270, h: 36, size: 28, color, bold: true });
    addText(s, t, { x: x + 24, y: 338, w: 270, h: 52, size: 24, color: C.white, bold: true });
    addText(s, b, { x: x + 24, y: 416, w: 270, h: 72, size: 17, color: C.text });
  });
  addFooter(s, 5);
  note(5, "Stage roadmap", "Stage 1 是 zero-training，用 external evaluator。Stage 2 是把 evaluator/localizer 学出来。Stage 3 才是更系统地迁移到不同 masked/discrete generator。");
}

// 6
{
  const s = baseSlide();
  addTitle(s, "Stage-1 position: ASCR already helps without training", "Mainline is Lumina-DiMOO; Show-o/MMaDA remain comparison lines.");
  addBox(s, { x: 86, y: 236, w: 520, h: 326, fill: C.panel2, line: C.line });
  addText(s, "Clean score summary", { x: 116, y: 266, w: 390, h: 34, size: 24, color: C.white, bold: true });
  addMiniBar(s, "Lumina baseline", 82.8, 100, 120, 338, 280, C.cyan2);
  addMiniBar(s, "Lumina + ASCR", 84.4, 100, 120, 390, 280, C.green);
  addMiniBar(s, "Show-o baseline", 73.4, 100, 120, 442, 280, C.cyan2);
  addMiniBar(s, "Show-o + ASCR", 78.1, 100, 120, 494, 280, C.green);
  addBox(s, { x: 684, y: 236, w: 420, h: 326, fill: C.panel2, line: C.green });
  addMetric(s, "Lumina ASCR effect", "+1.6pp", "82.8% → 84.4%", { x: 720, y: 286, w: 340, h: 120, accent: C.green });
  addMetric(s, "Show-o ASCR effect", "+4.7pp", "73.4% → 78.1%", { x: 720, y: 426, w: 340, h: 120, accent: C.amber });
  addFooter(s, 6, "Source: README empirical summary");
  note(6, "Stage-1 results", "Stage 1 已经证明 ASCR 不只是工程结构。Lumina 上提升比较小但没有 regression，因为 base model 本身强；Show-o 上提升更明显，说明 semantic reopening 在弱一些的 generator 上空间更大。");
}

// 7
{
  const s = baseSlide();
  addTitle(s, "What exactly gets distilled?", "The student learns the evaluator/localizer role, not the selector.");
  addBox(s, { x: 100, y: 242, w: 455, h: 260, fill: C.panel2, line: C.amber });
  addText(s, "semantic localizer / evaluator", { x: 130, y: 275, w: 390, h: 32, size: 25, color: C.amber, bold: true });
  addText(s, "Input: prompt + current grid image\nOutput: has_error + grid cells + correction instruction", { x: 130, y: 340, w: 380, h: 88, size: 20, color: C.text });
  addText(s, "This is the distillation target.", { x: 130, y: 452, w: 350, h: 30, size: 19, color: C.white, bold: true });
  addBox(s, { x: 718, y: 242, w: 455, h: 260, fill: C.panel2, line: C.cyan });
  addText(s, "GridSemanticReopeningSelector", { x: 748, y: 275, w: 390, h: 32, size: 25, color: C.cyan, bold: true });
  addText(s, "Input: selected grid cells\nOutput: token reopen mask", { x: 748, y: 350, w: 370, h: 68, size: 21, color: C.text });
  addText(s, "This stays fixed for now.", { x: 748, y: 452, w: 350, h: 30, size: 19, color: C.white, bold: true });
  addFooter(s, 7);
  note(7, "What is distilled", "这里我特别区分两个概念。semantic localizer/evaluator 是看 prompt 和 image，判断有没有 semantic error、错在哪些 grid cells。selector 只是把 grid cells 映射成 token reopen mask。所以真正要 distill 的是 localizer/evaluator，不是 selector。");
}

// 8
{
  const s = baseSlide();
  addTitle(s, "Before/after benchmark design", "The after image comes from the full ASCR loop, not a single forced reopen.");
  addText(s, "before distill", { x: 100, y: 236, w: 240, h: 28, size: 22, color: C.cyan, bold: true });
  addPipelineNode(s, "prompt", "", 100, 292, 150, C.cyan);
  addArrow(s, 268, 324, 70, 34);
  addPipelineNode(s, "generator", "initial image", 360, 292, 190, C.green);
  addText(s, "after distill", { x: 100, y: 438, w: 240, h: 28, size: 22, color: C.amber, bold: true });
  addPipelineNode(s, "prompt", "", 100, 494, 120, C.cyan);
  addArrow(s, 238, 526, 54, 34);
  addPipelineNode(s, "generator", "same base model", 310, 494, 150, C.green);
  addArrow(s, 478, 526, 54, 34);
  addPipelineNode(s, "student localizer", "learned evaluator", 548, 494, 190, C.amber);
  addArrow(s, 756, 526, 54, 34);
  addPipelineNode(s, "selector", "cells → mask", 828, 494, 140, C.cyan);
  addArrow(s, 986, 526, 54, 34);
  addPipelineNode(s, "ASCR loop", "0..max_iterations", 1058, 494, 170, C.green);
  addBox(s, { x: 650, y: 270, w: 500, h: 120, fill: C.panel3, line: C.line });
  addText(s, "Qwen3.7 judge compares initial image vs raw final ASCR candidate.", { x: 690, y: 304, w: 420, h: 54, size: 22, color: C.white, bold: true, align: "center" });
  addFooter(s, 8);
  note(8, "Before/after design", "before distill 是 prompt 到 generator 得到 initial image。after distill 是同一个 generator 加上 student localizer，再经过 existing GridSemanticReopeningSelector 和 ASCR loop，最终得到 revised image。这里不是固定 reopen 一次，而是 loop 自己决定，最多到 max_iterations。");
}

// 9
{
  const s = baseSlide();
  addTitle(s, "Teacher data: Qwen3.7 compact supervision", "Teacher gives both quality labels and semantic localization labels.");
  addMetric(s, "Quality labels", "64", "before/final pairwise scoring", { x: 120, y: 250, w: 300, h: 140, accent: C.green });
  addMetric(s, "Localization labels", "79", "prompt + grid image → cells", { x: 490, y: 250, w: 300, h: 140, accent: C.amber });
  addMetric(s, "Unresolved errors", "0", "after repair/rerun", { x: 860, y: 250, w: 300, h: 140, accent: C.cyan });
  addBox(s, { x: 205, y: 470, w: 870, h: 80, fill: C.panel3, line: C.line });
  addText(s, "Operational split: login node handles OFOX/Qwen API; GPU compute nodes handle generation and training.", { x: 240, y: 496, w: 800, h: 32, size: 22, color: C.white, bold: true, align: "center" });
  addFooter(s, 9, "Source: AI_COLLAB_LOG and teacher distillation artifacts");
  note(9, "Teacher data", "我用 Qwen3.7-plus through OFOX 做 teacher。它给两类 supervision：一个是 quality comparison，一个是 localization labels。现在 hard64 compact dataset 已经 clean 到没有 unresolved errors。");
}

// 10
{
  const s = baseSlide();
  addTitle(s, "Student localizer v0", "First end-to-end student baseline: meaningful, but intentionally lightweight.");
  addPipelineNode(s, "prompt", "hashed text features", 90, 324, 170, C.cyan);
  addArrow(s, 278, 356, 68, 34);
  addPipelineNode(s, "grid image", "RGB cell features", 365, 324, 180, C.green);
  addArrow(s, 563, 356, 68, 34);
  addPipelineNode(s, "v0 student", "centroid-style scoring", 650, 324, 210, C.amber);
  addArrow(s, 878, 356, 68, 34);
  addPipelineNode(s, "SemanticEvaluation", "has_error + cells", 965, 324, 230, C.cyan);
  addBullets(s, [
    "v0 is not cell-prior: it reads prompt and grid image.",
    "Still shallow: no neural multimodal representation yet.",
    "Purpose: validate the Stage-2 measurement and deployment path."
  ], 120, 500, 930, 40, 20);
  addFooter(s, 10);
  note(10, "Student v0", "v0 是第一个真正的 student localizer baseline。它不是 cell-prior；它至少读 prompt 和 grid image。但它仍然是 shallow features，不是 neural model。它的意义是验证整个 Stage-2 pipeline 能跑通。");
}

// 11
{
  const s = baseSlide();
  addTitle(s, "v0 quantitative results", "Pipeline works; the first student remains weak.");
  addMetric(s, "Train rows", String(metrics.train_rows), "localization examples", { x: 90, y: 232, w: 240, h: 120, accent: C.cyan });
  addMetric(s, "Eval rows", String(metrics.eval_rows), "holdout examples", { x: 370, y: 232, w: 240, h: 120, accent: C.cyan });
  addMetric(s, "Hit-any", pct(metrics.eval.hit_any_rate, 1), "2 / 10 evaluated", { x: 650, y: 232, w: 240, h: 120, accent: C.amber });
  addMetric(s, "Mean F1", n(metrics.eval.mean_f1, 2), "cell localization", { x: 930, y: 232, w: 240, h: 120, accent: C.amber });
  addBox(s, { x: 150, y: 430, w: 980, h: 120, fill: C.panel2, line: C.line });
  addText(s, "Interpretation: v0 is a reproducible weak baseline, not a final Stage-2 student.", { x: 190, y: 462, w: 900, h: 38, size: 26, color: C.white, bold: true, align: "center" });
  addText(s, "The next useful step is improving the student data/model, not rerunning v0 unchanged.", { x: 210, y: 506, w: 860, h: 28, size: 18, color: C.muted, align: "center" });
  addFooter(s, 11, "Source: outputs/stage2_students/grid_localizer_v0/metrics.json");
  note(11, "v0 quantitative results", "结果很诚实：pipeline works, but student is weak。Hit-any 只有 0.2，mean F1 只有 0.13。这说明现在不是 rerun v0，而是要 improve student data and model。");
}

// 12
{
  const s = baseSlide();
  addTitle(s, "Before/after image benchmark results", "No regression is good; meaningful gains require a stronger student.");
  addBox(s, { x: 92, y: 230, w: 500, h: 320, fill: C.panel2, line: C.cyan });
  addText(s, "In-domain hard64 holdout", { x: 122, y: 262, w: 420, h: 34, size: 24, color: C.cyan, bold: true });
  addMetric(s, "Rows", String(judgeIn.row_count), "Qwen3.7 judged", { x: 126, y: 322, w: 180, h: 106, accent: C.cyan });
  addMetric(s, "Winner", "1 after", "16 ties, 0 before", { x: 326, y: 322, w: 210, h: 106, accent: C.green });
  addMetric(s, "Mean delta", `+${n(judgeIn.mean_delta_after_minus_before, 3)}`, "after - before", { x: 226, y: 440, w: 230, h: 92, accent: C.green });
  addBox(s, { x: 688, y: 230, w: 500, h: 320, fill: C.panel2, line: C.amber });
  addText(s, "Geneval smoke16", { x: 718, y: 262, w: 420, h: 34, size: 24, color: C.amber, bold: true });
  addMetric(s, "Rows", String(judgeGe.row_count), "out-domain smoke", { x: 722, y: 322, w: 180, h: 106, accent: C.amber });
  addMetric(s, "Winner", "all tie", "0 after, 0 before", { x: 922, y: 322, w: 210, h: 106, accent: C.cyan });
  addMetric(s, "Mean delta", n(judgeGe.mean_delta_after_minus_before, 3), "after - before", { x: 822, y: 440, w: 230, h: 92, accent: C.cyan });
  addFooter(s, 12, "Source: outputs/api_judges/student_localizer_v0/*/summary.json");
  note(12, "Image benchmark results", "In-domain 有非常小的 positive delta，Geneval 全 tie。这个结果不是失败，它说明 measurement loop 是健康的，没有明显 regression，但要看到实质提升，需要更强的 student localizer。");
}

// 13
{
  const s = baseSlide();
  addTitle(s, "Engineering lessons that changed the evaluation", "Several small bugs would have produced misleading science if left unfixed.");
  const lessons = [
    ["Prompt truncation", "Slurm LIMIT=16 truncated in-domain holdout from 17 to 16.", C.red],
    ["Fallback semantics", "final_decoded_image can become initial image under return_initial_on_max_error.", C.amber],
    ["Judge reruns", "--overwrite needed reset, dedupe, and stale-error pruning.", C.cyan],
    ["API isolation", "Compute nodes must blank OFOX env vars; API stays on login node.", C.green],
  ];
  lessons.forEach(([t, b, color], i) => {
    const x = 90 + (i % 2) * 575;
    const y = 235 + Math.floor(i / 2) * 165;
    addBox(s, { x, y, w: 500, h: 128, fill: C.panel2, line: color });
    addText(s, t, { x: x + 24, y: y + 24, w: 420, h: 28, size: 23, color, bold: true });
    addText(s, b, { x: x + 24, y: y + 64, w: 438, h: 46, size: 17, color: C.text });
  });
  addFooter(s, 13, "Source: AI_COLLAB_LOG server audit");
  note(13, "Engineering lessons", "这一步也暴露了几个关键工程问题。第一，LIMIT=16 差点截断 in-domain holdout。第二，fallback 会让 after image 变成 before image，所以必须记录 raw final candidate。第三，API judge rerun 要 overwrite and dedupe，否则会混旧结果。第四，compute node 不能继承 OFOX key。");
}

// 14
{
  const s = baseSlide();
  addTitle(s, "Student localizer v1: data and model upgrade", "Use benchmark images as additional teacher-labeled training data.");
  addPipelineNode(s, "v0 manifests", "before/after grid images", 80, 318, 190, C.cyan);
  addArrow(s, 288, 350, 58, 34);
  addPipelineNode(s, "Qwen3.7 teacher", "new localization labels", 360, 318, 210, C.amber);
  addArrow(s, 588, 350, 58, 34);
  addPipelineNode(s, "merged dataset", "Hard64 + in-domain + Geneval", 660, 318, 230, C.green);
  addArrow(s, 908, 350, 58, 34);
  addPipelineNode(s, "grid-localizer-v1", "linear model + richer features", 980, 318, 220, C.cyan);
  addBullets(s, [
    "Richer features: RGB mean/std, contrast, neighbor statistics, prompt hash, domain metadata.",
    "Still lightweight and reproducible: JSON model, no neural checkpoint yet.",
    "Purpose: decide whether shallow student modeling is enough before moving to neural Stage 2."
  ], 112, 502, 1020, 40, 18);
  addFooter(s, 14);
  note(14, "v1 plan", "v1 的核心是增加 supervision。我们会让 Qwen3.7 teacher 对 v0 benchmark 产生的 before/after grid images 再做 localization，然后合并 hard64、in-domain、Geneval labels，训练 grid-localizer-v1。v1 还是 lightweight，但比 v0 有更多 image statistics 和 domain features。");
}

// 15
{
  const s = baseSlide();
  addTitle(s, "Next milestone and decision point", "Run v1, compare to v0, then decide whether neural Stage 2 is justified.");
  addBox(s, { x: 92, y: 236, w: 520, h: 315, fill: C.panel2, line: C.green });
  addText(s, "Next milestone", { x: 124, y: 266, w: 420, h: 32, size: 25, color: C.green, bold: true });
  addBullets(s, [
    "Train grid-localizer-v1 on merged teacher labels.",
    "Run in-domain and Geneval before/after benchmarks.",
    "Judge with Qwen3.7 and compare v1 vs v0."
  ], 126, 330, 420, 48, 20);
  addBox(s, { x: 688, y: 236, w: 500, h: 315, fill: C.panel2, line: C.amber });
  addText(s, "Decision after v1", { x: 720, y: 266, w: 420, h: 32, size: 25, color: C.amber, bold: true });
  addBullets(s, [
    "If v1 improves: scale labels and benchmark size.",
    "If v1 remains weak: move to neural/multimodal student localizer.",
    "Keep selector fixed until evaluator/localizer is strong."
  ], 722, 330, 410, 60, 18);
  addFooter(s, 15);
  note(15, "Next milestone", "下一步我会用同样 benchmark 比较 v1 和 v0：winner counts、mean score delta、fallback count、changed image count。如果 v1 仍然弱，那就说明 shallow localizer 不够，下一步应该进入 neural localizer 或更强 multimodal student。");
}

function slideGlow(slide) {
  slide.shapes.add({
    geometry: "arc",
    position: { left: 880, top: 90, width: 420, height: 420 },
    fill: "none",
    line: { style: "solid", fill: C.cyan2, width: 4, transparency: 60 },
  });
  slide.shapes.add({
    geometry: "arc",
    position: { left: 940, top: 150, width: 300, height: 300 },
    fill: "none",
    line: { style: "solid", fill: C.amber, width: 3, transparency: 55 },
  });
}

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(QA_DIR, { recursive: true });

await fs.writeFile(
  NOTES_TXT,
  "\uFEFF" + [
    "ASCR Stage-2 Progress Advisor Meeting - Speaker Notes",
    "Style: 中文讲解为主，关键术语保留 English。",
    "",
    ...notes,
  ].join("\n"),
  "utf8",
);

for (const [index, slide] of p.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await p.export({ slide, format: "png", scale: 1 });
  await writeBlob(path.join(QA_DIR, `${stem}.png`), png);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
}

const montage = await p.export({ format: "webp", montage: true, scale: 1 });
await writeBlob(path.join(QA_DIR, "deck-montage.webp"), montage);

const pptx = await PresentationFile.exportPptx(p);
await pptx.save(FINAL_PPTX);

console.log(JSON.stringify({
  pptx: FINAL_PPTX,
  notes: NOTES_TXT,
  slides: p.slides.items.length,
  preview: path.join(QA_DIR, "deck-montage.webp"),
}, null, 2));
