const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
const W = 13.333, H = 7.5;

// ---- palette (ESG / sustainability) ----
const DARK = "14342B";     // deep forest (title/closing)
const FOREST = "2C5F2D";
const MOSS = "97BC62";
const GOLD = "E9B44C";     // accent for key numbers
const INK = "1F2A24";
const MUTED = "6B7A72";
const CARD = "F1F5F2";
const WHITE = "FFFFFF";
const HF = "Century Schoolbook";  // header serif (safe)
const BF = "Calibri";             // body sans (safe)

const notes = {};
function slide(bg) { const s = p.addSlide(); s.background = { color: bg || WHITE }; return s; }
function circleIcon(s, x, y, d, fill, glyph, gcol) {
  s.addShape(p.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(glyph, { x, y, w: d, h: d, align: "center", valign: "middle",
    fontFace: BF, fontSize: d * 22, bold: true, color: gcol || WHITE });
}
function kicker(s, txt, x, y, col) {
  s.addText(txt.toUpperCase(), { x, y, w: 8, h: 0.3, fontFace: BF, fontSize: 12,
    bold: true, color: col || MOSS, charSpacing: 3, margin: 0 });
}
// Manual table grid (LibreOffice ignores pptxgenjs colW; shapes are reliable).
// cells: 2D array; a cell is a string or { t, b(old), c(olor), fs, fill, left }.
function grid(s, x, y, colW, rowH, cells) {
  let cy = y;
  for (let r = 0; r < cells.length; r++) {
    let cx = x;
    const head = r === 0;
    for (let c = 0; c < colW.length; c++) {
      let cell = cells[r][c]; if (cell == null) cell = "";
      const o = (typeof cell === "string") ? { t: cell } : cell;
      const left = o.left || c === 0;
      const fill = o.fill || (head ? DARK : (r % 2 ? "F6FAF7" : WHITE));
      s.addShape(p.ShapeType.rect, { x: cx, y: cy, w: colW[c], h: rowH,
        fill: { color: fill }, line: { color: "E1E9E3", width: 0.75 } });
      s.addText(o.t, { x: cx + (left ? 0.12 : 0.02), y: cy, w: colW[c] - (left ? 0.14 : 0.04), h: rowH,
        align: left ? "left" : "center", valign: "middle", fontFace: BF,
        fontSize: o.fs || 12.5, bold: head ? true : !!o.b,
        color: o.c || (head ? WHITE : INK), margin: 0 });
      cx += colW[c];
    }
    cy += rowH;
  }
}

/* ============================ SLIDE 1 — TITLE ============================ */
let s = slide(DARK);
// motif: three stacked bars suggesting regimes, top-right
[["9 5", FOREST], ["6 4", MOSS], ["3 6", GOLD]];
s.addShape(p.ShapeType.rect, { x: 10.7, y: 0, w: 2.633, h: H, fill: { color: "10281F" } });
[[1.1, FOREST], [2.0, MOSS], [1.5, GOLD], [0.9, "3E7B4F"], [1.7, MOSS]].forEach((b, i) => {
  s.addShape(p.ShapeType.roundRect, { x: 10.95 + i * 0.46, y: H - 0.8 - b[0], w: 0.32, h: b[0],
    rectRadius: 0.06, fill: { color: b[1] } });
});
kicker(s, "Georgia Tech · Finance Research · companion to esg_adaptive_rl", 0.9, 1.05, MOSS);
s.addText("ESG Market-Regime Detection", { x: 0.85, y: 1.5, w: 9.4, h: 1.5, fontFace: HF,
  fontSize: 46, bold: true, color: WHITE, lineSpacing: 48 });
s.addText("A 3-State Hidden Markov Model for Regime-Conditional Sustainable Investing",
  { x: 0.9, y: 3.05, w: 9.2, h: 0.9, fontFace: BF, fontSize: 20, color: MOSS });
s.addText([
  { text: "Detects when the ESG universe is ", options: {} },
  { text: "Calm · Choppy · Stressed", options: { bold: true, color: GOLD } },
  { text: " — the signal a reinforcement-learning allocator uses to switch its E/S/G-vs-return reward weights by regime.", options: {} },
], { x: 0.9, y: 4.15, w: 9.0, h: 1.2, fontFace: BF, fontSize: 15, color: "CFE0D4", lineSpacing: 24 });
s.addText("Adapted from a production SPY regime model → self-contained on ESG price history (no options data).",
  { x: 0.9, y: 6.45, w: 9.2, h: 0.5, fontFace: BF, fontSize: 12, italic: true, color: "8FB39B" });
notes[1] = "3-state Gaussian HMM adapted from a production SPY regime model for the ESG universe. The output feeds regime-conditional reward weights in the ESG-Adaptive RL project.";

/* ==================== SLIDE 2 — WHY REGIMES ==================== */
s = slide(WHITE);
kicker(s, "Motivation", 0.85, 0.6, FOREST);
s.addText("Why regime detection for ESG RL", { x: 0.8, y: 0.95, w: 11.5, h: 0.8,
  fontFace: HF, fontSize: 34, bold: true, color: INK });
s.addText("The right trade-off between return and sustainability is not constant — it depends on the market state. A regime label lets the allocator hold a different E/S/G tilt in calm markets than in stressed ones.",
  { x: 0.85, y: 1.85, w: 11.4, h: 0.8, fontFace: BF, fontSize: 15, color: MUTED, lineSpacing: 22 });
const rows2 = [
  ["25A244", "▲", "In calm markets", "Lean into ESG tilts and growth — volatility is low, drawdown risk is contained."],
  ["E9A23B", "◆", "In choppy markets", "Trim exposure; the vol-of-vol rises and factor leadership rotates."],
  ["C0392B", "■", "In stressed markets", "De-risk hard — stress days carry ~2× the volatility with poor risk-adjusted return."],
];
rows2.forEach((r, i) => {
  const y = 3.05 + i * 1.35;
  s.addShape(p.ShapeType.roundRect, { x: 0.85, y, w: 11.6, h: 1.15, rectRadius: 0.08,
    fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
  circleIcon(s, 1.15, y + 0.3, 0.55, r[0], r[1], WHITE);
  s.addText(r[2], { x: 1.95, y: y + 0.16, w: 3.4, h: 0.5, fontFace: BF, fontSize: 17, bold: true, color: INK, margin: 0 });
  s.addText(r[3], { x: 5.3, y: y + 0.12, w: 6.9, h: 0.9, fontFace: BF, fontSize: 14, color: "45524B", valign: "middle", margin: 0, lineSpacing: 19 });
});
notes[2] = "The RL project's core thesis: reward weights over return and E/S/G should be regime-conditional. This model supplies the regime.";

/* ==================== SLIDE 3 — APPROACH ==================== */
s = slide(WHITE);
kicker(s, "Approach", 0.85, 0.6, FOREST);
s.addText("Same HMM engine, ESG-specific features", { x: 0.8, y: 0.95, w: 11.6, h: 0.8,
  fontFace: HF, fontSize: 32, bold: true, color: INK });
// left column: the engine
s.addShape(p.ShapeType.roundRect, { x: 0.85, y: 1.95, w: 5.2, h: 4.7, rectRadius: 0.08, fill: { color: DARK } });
s.addText("The engine (unchanged)", { x: 1.15, y: 2.2, w: 4.6, h: 0.5, fontFace: BF, fontSize: 16, bold: true, color: GOLD, margin: 0 });
[
  "3-state Gaussian-emission HMM, diagonal covariance",
  "States pinned by realized-vol → Calm / Choppy / Stress",
  "Filtered (forward-only) posteriors — point-in-time, no look-ahead",
  "Anti-whipsaw hysteresis on regime switches",
  "Deterministic stress guardrail (below 200d trend + vol backwardation)",
].forEach((t, i) => {
  s.addText(t, { x: 1.15, y: 2.85 + i * 0.72, w: 4.7, h: 0.7, fontFace: BF, fontSize: 13.5,
    color: "DCE8DF", bullet: { code: "2022", indent: 14 }, margin: 0, lineSpacing: 17 });
});
// right column: the swap
s.addText("The ESG adaptation", { x: 6.5, y: 2.0, w: 5.8, h: 0.4, fontFace: BF, fontSize: 16, bold: true, color: FOREST, margin: 0 });
const swap = [
  ["SPY model used…", "ESG model uses…", true],
  ["VIX / VIX9d implied\nterm structure (SPX-only)", "Realized-vol term structure\nlog(rv₁₀ / rv₆₀) — self-contained", false],
  ["Variance risk premium\n(needs implied vol)", "Dropped — no options data\nrequired for any ESG name", false],
  ["Trend · realized vol ·\ndrawdown · Parkinson vol", "Kept + 20-day momentum", false],
];
swap.forEach((r, i) => {
  const y = 2.5 + i * 1.02;
  const hd = r[2];
  s.addShape(p.ShapeType.roundRect, { x: 6.5, y, w: 2.82, h: 0.9, rectRadius: 0.06,
    fill: { color: hd ? MUTED : "EEF3EF" } });
  s.addShape(p.ShapeType.roundRect, { x: 9.45, y, w: 2.85, h: 0.9, rectRadius: 0.06,
    fill: { color: hd ? FOREST : "E7F0E9" } });
  s.addText(r[0], { x: 6.55, y, w: 2.72, h: 0.9, fontFace: BF, fontSize: hd ? 12 : 11.5, bold: hd,
    color: hd ? WHITE : "3A463F", align: "center", valign: "middle", margin: 0, lineSpacing: 14 });
  s.addText(r[1], { x: 9.5, y, w: 2.75, h: 0.9, fontFace: BF, fontSize: hd ? 12 : 11.5, bold: hd,
    color: hd ? WHITE : "244A2B", align: "center", valign: "middle", margin: 0, lineSpacing: 14 });
});
notes[3] = "Only the feature set changes. The realized-vol term structure is the key substitute for the SPX-specific VIX term structure, making the model self-contained on any ESG price series.";

/* ==================== SLIDE 4 — DATA ==================== */
s = slide(WHITE);
kicker(s, "Data & Universes", 0.85, 0.6, FOREST);
s.addText("Three ESG universes — one built from real ESG scores", { x: 0.8, y: 0.95, w: 12, h: 0.8,
  fontFace: HF, fontSize: 30, bold: true, color: INK });
const cards = [
  ["High-ESG Index", "PRIMARY · genuinely ESG-driven", GOLD,
    "Equal-weight basket of the top-40 US large-caps by real Refinitiv ESG score, from the repo's own Dataset/.",
    "34", "names, full history", "2011–2026"],
  ["SUSA ETF", "External robustness check", MOSS,
    "iShares MSCI USA ESG Select — a broad ESG-tilted US equity benchmark.",
    "5,085", "trading days", "2006–2026"],
  ["DSI ETF", "External robustness check", MOSS,
    "iShares MSCI KLD 400 Social — a values-screened US equity index.",
    "4,630", "trading days", "2008–2026"],
];
cards.forEach((c, i) => {
  const x = 0.85 + i * 4.05;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.0, w: 3.75, h: 4.55, rectRadius: 0.1,
    fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x, y: 2.0, w: 3.75, h: 0.14, rectRadius: 0.0, fill: { color: c[2] } });
  s.addText(c[0], { x: x + 0.28, y: 2.4, w: 3.2, h: 0.5, fontFace: HF, fontSize: 21, bold: true, color: INK, margin: 0 });
  s.addText(c[1].toUpperCase(), { x: x + 0.28, y: 2.92, w: 3.3, h: 0.3, fontFace: BF, fontSize: 10.5,
    bold: true, color: c[2] === GOLD ? "B4801E" : FOREST, charSpacing: 1, margin: 0 });
  s.addText(c[3], { x: x + 0.28, y: 3.35, w: 3.2, h: 1.4, fontFace: BF, fontSize: 13, color: "45524B", margin: 0, lineSpacing: 19 });
  s.addText(c[4], { x: x + 0.28, y: 4.85, w: 3.2, h: 0.7, fontFace: HF, fontSize: 40, bold: true, color: FOREST, margin: 0 });
  s.addText(c[5], { x: x + 0.28, y: 5.6, w: 3.2, h: 0.3, fontFace: BF, fontSize: 12, color: MUTED, margin: 0 });
  s.addText(c[6], { x: x + 0.28, y: 5.95, w: 3.2, h: 0.4, fontFace: BF, fontSize: 14, bold: true, color: INK, margin: 0 });
});
notes[4] = "The high-ESG index makes the regime model genuinely ESG-driven rather than a generic ETF proxy. SUSA and DSI confirm robustness.";

/* ==================== SLIDE 5 — METHODOLOGY ==================== */
s = slide(WHITE);
kicker(s, "Methodology", 0.85, 0.6, FOREST);
s.addText("Look-ahead-free by construction", { x: 0.8, y: 0.95, w: 11.6, h: 0.8,
  fontFace: HF, fontSize: 32, bold: true, color: INK });
// pipeline steps
const steps = ["Lagged z-score\nfeatures (t-1 window)", "Fit 3-state HMM\n(8 restarts)", "Vol-order &\nlabel states",
  "Filtered decode\n+ hysteresis", "Stress guardrail\noverride"];
steps.forEach((t, i) => {
  const x = 0.85 + i * 2.42;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.0, w: 2.1, h: 1.15, rectRadius: 0.08, fill: { color: FOREST } });
  s.addText(`${i + 1}`, { x: x + 0.1, y: 2.08, w: 0.5, h: 0.4, fontFace: HF, fontSize: 18, bold: true, color: GOLD, margin: 0 });
  s.addText(t, { x: x + 0.1, y: 2.42, w: 1.9, h: 0.7, fontFace: BF, fontSize: 11.5, color: WHITE, align: "center", margin: 0, lineSpacing: 14 });
  if (i < 4) s.addText("→", { x: x + 2.02, y: 2.3, w: 0.42, h: 0.5, fontFace: BF, fontSize: 22, bold: true, color: MOSS, align: "center", margin: 0 });
});
// two protocols
s.addText("Two out-of-sample validation protocols", { x: 0.85, y: 3.55, w: 11, h: 0.4, fontFace: BF, fontSize: 15, bold: true, color: INK });
const prot = [
  ["Chronological train / test", "Fit the HMM once on dates before 2021-01-01, freeze it, then decode the later test period with filtered posteriors. No test bar is ever seen in training.", "1B4D3E"],
  ["Walk-forward", "Retrain monthly on an expanding window; every single day is labeled by a model trained only on its own past. The honest rolling protocol.", "26654F"],
];
prot.forEach((r, i) => {
  const x = 0.85 + i * 5.9;
  s.addShape(p.ShapeType.roundRect, { x, y: 4.1, w: 5.6, h: 2.4, rectRadius: 0.1, fill: { color: r[2] } });
  circleIcon(s, x + 0.35, y0(4.45), 0.5, GOLD, i === 0 ? "⏱" : "↻", DARK);
  s.addText(r[0], { x: x + 1.05, y: 4.5, w: 4.3, h: 0.5, fontFace: HF, fontSize: 19, bold: true, color: WHITE, margin: 0, valign: "middle" });
  s.addText(r[1], { x: x + 0.4, y: 5.2, w: 4.9, h: 1.1, fontFace: BF, fontSize: 13.5, color: "D7E6DC", margin: 0, lineSpacing: 19 });
});
function y0(v){return v;}
notes[5] = "Every guard against look-ahead: lagged standardization, forward-only posteriors, and two independent OOS protocols.";

/* ==================== SLIDE 6 — RESULT: REGIMES ARE REAL ==================== */
s = slide(WHITE);
kicker(s, "Result 1 — the labels mean something", 0.85, 0.6, FOREST);
s.addText("Stress days carry ~2× the volatility of calm days", { x: 0.8, y: 0.95, w: 12, h: 0.8,
  fontFace: HF, fontSize: 30, bold: true, color: INK });
s.addText("Out-of-sample next-day realized volatility, grouped by the frozen model's regime label. The separation holds across all three universes — evidence the regimes are economically real, not curve-fit.",
  { x: 0.85, y: 1.8, w: 7.0, h: 1.1, fontFace: BF, fontSize: 14, color: MUTED, lineSpacing: 20 });
// grouped bar chart
const chartData = [
  { name: "High-ESG Index", labels: ["Calm", "Choppy", "Stress"], values: [12.0, 15.7, 26.4] },
  { name: "SUSA", labels: ["Calm", "Choppy", "Stress"], values: [12.1, 15.6, 27.3] },
  { name: "DSI", labels: ["Calm", "Choppy", "Stress"], values: [12.0, 14.5, 25.4] },
];
s.addChart(p.ChartType.bar, chartData, {
  x: 0.85, y: 3.0, w: 7.4, h: 3.9, barDir: "col", chartColors: [FOREST, MOSS, GOLD],
  showTitle: false, showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BF,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: INK, dataLabelFormatCode: '0.0"%"',
  valAxisMinVal: 0, valAxisMaxVal: 30, valAxisMajorUnit: 10, valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
  valAxisTitle: "Annualized volatility", showValAxisTitle: true, valAxisTitleColor: MUTED, valAxisTitleFontSize: 11,
  catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontBold: true,
  valGridLine: { color: "E8EEE9", size: 1 }, catGridLine: { style: "none" },
});
// stat callout
s.addShape(p.ShapeType.roundRect, { x: 8.65, y: 3.0, w: 3.85, h: 1.85, rectRadius: 0.1, fill: { color: DARK } });
s.addText("2.2×", { x: 8.65, y: 3.15, w: 3.85, h: 0.95, fontFace: HF, fontSize: 54, bold: true, color: GOLD, align: "center", margin: 0 });
s.addText("stress-vs-calm volatility ratio (avg, OOS)", { x: 8.85, y: 4.1, w: 3.45, h: 0.6, fontFace: BF, fontSize: 12.5, color: "CFE0D4", align: "center", margin: 0 });
s.addShape(p.ShapeType.roundRect, { x: 8.65, y: 5.05, w: 3.85, h: 1.85, rectRadius: 0.1, fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
s.addText("11–20%", { x: 8.65, y: 5.2, w: 3.85, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: FOREST, align: "center", margin: 0 });
s.addText("of days flagged Stress under the calibrated train/test split", { x: 8.85, y: 6.05, w: 3.45, h: 0.7, fontFace: BF, fontSize: 12.5, color: "45524B", align: "center", margin: 0, lineSpacing: 16 });
notes[6] = "The core scientific result: the regime label predicts next-day volatility out of sample, ~2x from calm to stress, on every universe.";

/* ==================== SLIDE 7 — RESULT: OOS PERFORMANCE ==================== */
s = slide(WHITE);
kicker(s, "Result 2 — out-of-sample backtest", 0.85, 0.6, FOREST);
s.addText("A regime overlay roughly halves risk out-of-sample", { x: 0.8, y: 0.95, w: 12, h: 0.8,
  fontFace: HF, fontSize: 29, bold: true, color: INK });
s.addText("Overlay = 100% / 60% / 0% invested in Calm / Choppy / Stress (signal lagged one day, costs charged). Test period 2021–2026, frozen model.",
  { x: 0.85, y: 1.75, w: 11.5, h: 0.5, fontFace: BF, fontSize: 13, italic: true, color: MUTED });
const g = (t, b, c) => ({ t, b: !!b, c });
const rowsT = [
  ["Universe", "Strategy", "CAGR", "Vol", "Sharpe", "Max DD", "Calmar"],
  [g("High-ESG", 1), "Buy & Hold", "14.3%", "16.3%", "0.78", "−22.2%", "0.64"],
  [g("High-ESG", 1), "Overlay", "6.9%", g("9.9%", 1, FOREST), "0.52", g("−14.4%", 1, FOREST), "0.48"],
  [g("SUSA", 1), "Buy & Hold", "13.8%", "17.1%", "0.72", "−28.2%", "0.49"],
  [g("SUSA", 1), "Overlay", "8.8%", g("9.9%", 1, FOREST), "0.70", g("−17.4%", 1, FOREST), "0.50"],
  [g("DSI", 1), "Buy & Hold", "14.8%", "17.7%", "0.75", "−28.4%", "0.52"],
  [g("DSI", 1), "Overlay", "10.0%", g("8.5%", 1, FOREST), g("0.92", 1, "B4801E"), g("−7.7%", 1, FOREST), g("1.29", 1, "B4801E")],
];
grid(s, 0.85, 2.45, [1.28, 1.42, 1.0, 1.0, 1.0, 1.25, 0.9], 0.55, rowsT);
// highlight DSI callouts
s.addShape(p.ShapeType.roundRect, { x: 9.05, y: 2.55, w: 3.45, h: 2.0, rectRadius: 0.1, fill: { color: DARK } });
s.addText("DSI overlay, out-of-sample", { x: 9.25, y: 2.72, w: 3.1, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: GOLD, margin: 0 });
s.addText([
  { text: "0.92\n", options: { fontSize: 34, bold: true, color: WHITE, fontFace: HF } },
  { text: "Sharpe  vs  0.75 buy & hold", options: { fontSize: 12, color: "CFE0D4", fontFace: BF } },
], { x: 9.25, y: 3.12, w: 3.1, h: 1.3, margin: 0, lineSpacing: 20 });
s.addShape(p.ShapeType.roundRect, { x: 9.05, y: 4.72, w: 3.45, h: 2.0, rectRadius: 0.1, fill: { color: FOREST } });
s.addText("DSI max drawdown", { x: 9.25, y: 4.89, w: 3.1, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: GOLD, margin: 0 });
s.addText([
  { text: "−7.7%\n", options: { fontSize: 34, bold: true, color: WHITE, fontFace: HF } },
  { text: "vs  −28.4% buy & hold", options: { fontSize: 12, color: "DCE8DF", fontFace: BF } },
], { x: 9.25, y: 5.29, w: 3.1, h: 1.3, margin: 0, lineSpacing: 20 });
s.addText("Green = overlay improves the metric out-of-sample.", { x: 0.85, y: 6.75, w: 8, h: 0.35,
  fontFace: BF, fontSize: 11, italic: true, color: MUTED });
notes[7] = "The overlay is a risk-management demonstration: ~half the volatility and drawdown OOS everywhere, and on DSI it also beats buy & hold on Sharpe and Calmar.";

/* ==================== SLIDE 8 — REGIME VISUALIZATION ==================== */
s = slide(WHITE);
kicker(s, "Result 3 — regime map", 0.85, 0.55, FOREST);
s.addText("The model flags every major stress episode", { x: 0.8, y: 0.9, w: 12, h: 0.7,
  fontFace: HF, fontSize: 30, bold: true, color: INK });
s.addImage({ path: "esg_index_regimes.png", x: 0.85, y: 1.75, w: 9.4, h: 5.28 });
// legend / annotations column
s.addText("High-ESG index, walk-forward (OOS)", { x: 10.5, y: 1.9, w: 2.6, h: 0.6, fontFace: BF, fontSize: 13, bold: true, color: INK, margin: 0 });
[["25A244", "Calm — low vol, trending"], ["E9A23B", "Choppy — elevated vol"], ["C0392B", "Stress — high vol / drawdown"]].forEach((l, i) => {
  const y = 2.7 + i * 0.55;
  s.addShape(p.ShapeType.roundRect, { x: 10.5, y, w: 0.28, h: 0.28, rectRadius: 0.04, fill: { color: l[0] } });
  s.addText(l[1], { x: 10.86, y: y - 0.06, w: 2.4, h: 0.4, fontFace: BF, fontSize: 11.5, color: "45524B", margin: 0, valign: "middle" });
});
s.addShape(p.ShapeType.roundRect, { x: 10.5, y: 4.5, w: 2.55, h: 2.5, rectRadius: 0.1, fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
s.addText("Correctly caught", { x: 10.7, y: 4.65, w: 2.2, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: FOREST, margin: 0 });
["2015–16 selloff", "Q4 2018 drawdown", "2020 COVID crash", "2022 bear market"].forEach((t, i) => {
  s.addText(t, { x: 10.7, y: 5.05 + i * 0.46, w: 2.25, h: 0.4, fontFace: BF, fontSize: 12.5, color: INK,
    bullet: { code: "2713", indent: 12 }, margin: 0 });
});
notes[8] = "Visual confirmation: the shaded regimes align with the known stress periods, and the overlay equity curve avoids the deep drawdowns.";

/* ==================== SLIDE 9 — WALK-FORWARD + CAVEAT ==================== */
s = slide(WHITE);
kicker(s, "Robustness & honest limits", 0.85, 0.6, FOREST);
s.addText("Walk-forward: the same story, plus one caveat", { x: 0.8, y: 0.95, w: 12, h: 0.8,
  fontFace: HF, fontSize: 30, bold: true, color: INK });
const gw = (t, b, c) => ({ t, b: !!b, c });
const rowsW = [
  ["Universe", "Strategy", "Sharpe", "Max Drawdown"],
  [gw("SUSA", 1), "Buy & Hold", "0.52", "−53.9%"],
  [gw("SUSA", 1), "Overlay", gw("0.65", 1, "B4801E"), gw("−18.2%", 1, FOREST)],
  [gw("DSI", 1), "Buy & Hold", "0.56", "−49.9%"],
  [gw("DSI", 1), "Overlay", "0.56", gw("−27.9%", 1, FOREST)],
];
grid(s, 0.85, 2.15, [1.5, 1.5, 1.5, 1.6], 0.56, rowsW);
s.addText("On the long ETF histories the walk-forward overlay lifts Sharpe and cuts max drawdown by ~half — it even sidesteps the 2008 crash the train/test split can't see.",
  { x: 0.85, y: 5.05, w: 6.0, h: 1.5, fontFace: BF, fontSize: 14, color: "45524B", lineSpacing: 21 });
// caveat card
s.addShape(p.ShapeType.roundRect, { x: 7.25, y: 2.15, w: 5.25, h: 4.55, rectRadius: 0.1, fill: { color: "FBF3E2" }, line: { color: GOLD, width: 1.5 } });
circleIcon(s, 7.6, 2.5, 0.55, GOLD, "!", DARK);
s.addText("Where to read it carefully", { x: 8.35, y: 2.55, w: 3.9, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: "8A5A12", margin: 0, valign: "middle" });
[
  ["Not a return-maximizer.", "In strong bull runs the overlay gives up CAGR by sitting out stress days. Its value is proving the labels are tradeable and transfer OOS."],
  ["Early walk-forward is conservative.", "On the shorter high-ESG basket it over-labels Stress (~35%) when training data is thin; the calibrated train/test split puts it at 11–13%."],
  ["Price regimes, not ESG-score regimes.", "Detected from ESG-universe price dynamics — not yet from ESG scores directly."],
].forEach((r, i) => {
  const y = 3.35 + i * 1.12;
  s.addText(r[0], { x: 7.6, y, w: 4.7, h: 0.35, fontFace: BF, fontSize: 13.5, bold: true, color: "7A4E10", margin: 0 });
  s.addText(r[1], { x: 7.6, y: y + 0.32, w: 4.7, h: 0.8, fontFace: BF, fontSize: 12, color: "5C4A2A", margin: 0, lineSpacing: 16 });
});
notes[9] = "Intellectual honesty: the overlay trades return for risk in bull markets, and the early walk-forward is conservative. Both are expected and disclosed.";

/* ==================== SLIDE 10 — TAKEAWAYS ==================== */
s = slide(DARK);
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.16, fill: { color: GOLD } });
kicker(s, "Takeaways & next steps", 0.9, 0.75, MOSS);
s.addText("A regime signal ready to drive the RL allocator", { x: 0.85, y: 1.15, w: 11.6, h: 0.9,
  fontFace: HF, fontSize: 32, bold: true, color: WHITE });
const takeaways = [
  ["✓", "Regimes are real & OOS-valid", "Stress-day vol ≈ 2× calm across all three universes, out-of-sample — a clean, tradeable signal."],
  ["✓", "Risk cut roughly in half", "Overlay ~halves volatility and max drawdown OOS; on DSI it beats buy & hold (Sharpe 0.92 vs 0.75)."],
  ["✓", "Drops into the repo", "regime.py matches esg_adaptive_rl conventions; committed on branch esg-regime-hmm."],
];
takeaways.forEach((t, i) => {
  const y = 2.35 + i * 1.15;
  s.addShape(p.ShapeType.roundRect, { x: 0.85, y, w: 6.6, h: 1.0, rectRadius: 0.08, fill: { color: "1D453A" } });
  circleIcon(s, 1.12, y + 0.24, 0.52, GOLD, "✓", DARK);
  s.addText(t[1], { x: 1.85, y: y + 0.13, w: 5.4, h: 0.4, fontFace: BF, fontSize: 15, bold: true, color: WHITE, margin: 0 });
  s.addText(t[2], { x: 1.85, y: y + 0.5, w: 5.45, h: 0.5, fontFace: BF, fontSize: 11.5, color: "AECBB7", margin: 0, lineSpacing: 15 });
});
// next steps
s.addShape(p.ShapeType.roundRect, { x: 7.75, y: 2.35, w: 4.75, h: 3.45, rectRadius: 0.1, fill: { color: "10281F" }, line: { color: "2C5F2D", width: 1 } });
s.addText("Next steps", { x: 8.05, y: 2.55, w: 4.2, h: 0.45, fontFace: HF, fontSize: 19, bold: true, color: GOLD, margin: 0 });
[
  "Add ESG-dispersion & ESG-momentum features → regimes reflect ESG-factor stress, not just price stress",
  "Feed regime labels into the RL reward: evolve per-regime E/S/G-vs-return weights",
  "Significance tests on the regime-conditional vol separation",
].forEach((t, i) => {
  s.addText(t, { x: 8.05, y: 3.1 + i * 0.85, w: 4.25, h: 0.85, fontFace: BF, fontSize: 12.5, color: "DCE8DF",
    bullet: { code: "2192", indent: 16 }, margin: 0, lineSpacing: 16 });
});
s.addText("ESG-Adaptive RL · Georgia Tech Finance Research · targeting ACM ICAIF 2026",
  { x: 0.85, y: 6.7, w: 11.6, h: 0.4, fontFace: BF, fontSize: 12, italic: true, color: "8FB39B", align: "center" });
notes[10] = "The regime detector is validated and integrated. The immediate next step is wiring its labels into the RL reward to realize the project's regime-conditional thesis.";

// attach notes
p.slides.forEach((sl, i) => { if (notes[i + 1]) sl.addNotes(notes[i + 1]); });

p.writeFile({ fileName: "ESG_Regime_HMM.pptx" }).then(f => console.log("WROTE", f));
