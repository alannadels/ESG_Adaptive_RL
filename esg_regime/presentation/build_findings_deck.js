const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
const W = 13.333, H = 7.5;

const DARK = "14342B", FOREST = "2C5F2D", MOSS = "97BC62", GOLD = "E9B44C";
const INK = "1F2A24", MUTED = "6B7A72", CARD = "F1F5F2", WHITE = "FFFFFF";
const RED = "B0413E", AMBER = "D08C22";
const HF = "Century Schoolbook", BF = "Calibri";
const notes = {};

function slide(bg) { const s = p.addSlide(); s.background = { color: bg || WHITE }; return s; }
function circleIcon(s, x, y, d, fill, glyph, gcol) {
  s.addShape(p.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(glyph, { x, y, w: d, h: d, align: "center", valign: "middle",
    fontFace: BF, fontSize: d * 22, bold: true, color: gcol || WHITE });
}
function kicker(s, t, x, y, col) {
  s.addText(t.toUpperCase(), { x, y, w: 9, h: 0.3, fontFace: BF, fontSize: 12, bold: true,
    color: col || MOSS, charSpacing: 3, margin: 0 });
}
function title(s, t, y) {
  s.addText(t, { x: 0.8, y: y || 0.95, w: 11.9, h: 0.8, fontFace: HF, fontSize: 30, bold: true, color: INK });
}
function grid(s, x, y, colW, rowH, cells, fs) {
  let cy = y;
  for (let r = 0; r < cells.length; r++) {
    let cx = x; const head = r === 0;
    for (let c = 0; c < colW.length; c++) {
      let cell = cells[r][c]; if (cell == null) cell = "";
      const o = (typeof cell === "string") ? { t: cell } : cell;
      const left = o.left || c === 0;
      const fill = o.fill || (head ? DARK : (r % 2 ? "F6FAF7" : WHITE));
      s.addShape(p.ShapeType.rect, { x: cx, y: cy, w: colW[c], h: rowH,
        fill: { color: fill }, line: { color: "E1E9E3", width: 0.75 } });
      s.addText(o.t, { x: cx + (left ? 0.11 : 0.02), y: cy, w: colW[c] - (left ? 0.13 : 0.04), h: rowH,
        align: left ? "left" : "center", valign: "middle", fontFace: BF,
        fontSize: o.fs || fs || 12, bold: head ? true : !!o.b,
        color: o.c || (head ? WHITE : INK), margin: 0 });
      cx += colW[c];
    }
    cy += rowH;
  }
}

/* ---------------------------------------------------------------- 1 TITLE */
let s = slide(DARK);
s.addShape(p.ShapeType.rect, { x: 10.6, y: 0, w: 2.733, h: H, fill: { color: "10281F" } });
[2.92, 2.31, 2.00, 1.68, 2.71].forEach((v, i) => {
  const h = v * 0.62;
  s.addShape(p.ShapeType.roundRect, { x: 10.95 + i * 0.46, y: H - 0.9 - h, w: 0.32, h,
    rectRadius: 0.06, fill: { color: [MOSS, FOREST, GOLD, "3E7B4F", MOSS][i] } });
});
kicker(s, "Georgia Tech · Finance Research · ESG-Adaptive RL", 0.9, 1.0, MOSS);
s.addText("Do ESG Markets Have Regimes?", { x: 0.85, y: 1.45, w: 9.3, h: 1.4,
  fontFace: HF, fontSize: 44, bold: true, color: WHITE, lineSpacing: 46 });
s.addText("Findings from a 12-index hidden-Markov study of the ESG universe",
  { x: 0.9, y: 3.0, w: 9.2, h: 0.8, fontFace: BF, fontSize: 20, color: MOSS });
s.addText([
  { text: "Yes — and the effect is universal. ", options: { bold: true, color: GOLD } },
  { text: "All 12 ESG universes tested separate volatility states out-of-sample, and a regime overlay cut drawdowns on 11 of 11 published indices.", options: {} },
], { x: 0.9, y: 4.1, w: 9.0, h: 1.2, fontFace: BF, fontSize: 15, color: "CFE0D4", lineSpacing: 24 });
s.addText("A 3-state Gaussian HMM · walk-forward validated · 2005–2026",
  { x: 0.9, y: 6.5, w: 9.2, h: 0.4, fontFace: BF, fontSize: 12, italic: true, color: "8FB39B" });
notes[1] = "Headline: ESG market regimes are real and universal across ESG index styles.";

/* ------------------------------------------------------- 2 THE QUESTION */
s = slide(WHITE);
kicker(s, "The question", 0.85, 0.6, FOREST);
title(s, "Does the regime model need a bespoke ESG universe?");
s.addText("We first built a universe from real ESG scores — the top-40 US large-caps by Refinitiv ESG rating. Then we asked whether that bespoke construction was actually necessary, by running the identical model on 11 published ESG indices.",
  { x: 0.85, y: 1.8, w: 11.5, h: 0.9, fontFace: BF, fontSize: 15, color: MUTED, lineSpacing: 22 });
const q = [
  ["1", "Regime quality", "Does the HMM separate volatility states?", "Stress ÷ calm volatility ratio, out-of-sample", FOREST],
  ["2", "Economic value", "Are the regime labels tradeable?", "Overlay vs buy & hold: ΔSharpe, drawdown cut", "26654F"],
  ["3", "Universality", "Is there one ESG regime, or many?", "Cross-index label agreement on shared dates", "3E7B4F"],
];
q.forEach((r, i) => {
  const x = 0.85 + i * 3.93;
  s.addShape(p.ShapeType.roundRect, { x, y: 3.0, w: 3.63, h: 3.5, rectRadius: 0.1, fill: { color: r[4] } });
  circleIcon(s, x + 0.3, 3.3, 0.6, GOLD, r[0], DARK);
  s.addText(r[1], { x: x + 0.3, y: 4.05, w: 3.0, h: 0.45, fontFace: HF, fontSize: 20, bold: true, color: WHITE, margin: 0 });
  s.addText(r[2], { x: x + 0.3, y: 4.55, w: 3.05, h: 0.8, fontFace: BF, fontSize: 13.5, color: "DCE8DF", margin: 0, lineSpacing: 19 });
  s.addText("MEASURED BY", { x: x + 0.3, y: 5.45, w: 3.0, h: 0.25, fontFace: BF, fontSize: 9.5, bold: true, color: GOLD, charSpacing: 1, margin: 0 });
  s.addText(r[3], { x: x + 0.3, y: 5.72, w: 3.05, h: 0.7, fontFace: BF, fontSize: 12, color: "AECBB7", margin: 0, lineSpacing: 16 });
});
notes[2] = "Three test axes: regime quality, economic value, universality.";

/* --------------------------------------------------------- 3 DATA PANEL */
s = slide(WHITE);
kicker(s, "Data", 0.85, 0.6, FOREST);
title(s, "12 ESG universes spanning every major ESG style");
s.addText("Identical pipeline on all 12 — same features, same HMM, same walk-forward protocol, same costs. 1,498–5,085 trading days each, 2005–2026.",
  { x: 0.85, y: 1.78, w: 11.5, h: 0.5, fontFace: BF, fontSize: 14, color: MUTED });
const panel = [
  ["Universe", "Style", "History", "Bars"],
  [{ t: "esg_index", b: 1, c: "B4801E" }, "Score-constructed top-40 ESG basket", "2011–2026", "3,694"],
  [{ t: "SUSA", b: 1 }, "Broad ESG (MSCI USA ESG Select)", "2006–2026", "5,085"],
  [{ t: "DSI", b: 1 }, "Values-screened (KLD 400 Social)", "2008–2026", "4,630"],
  [{ t: "ERTH", b: 1 }, "Environmental / sustainable future", "2008–2026", "4,654"],
  [{ t: "ICLN", b: 1 }, "Clean-energy thematic", "2009–2026", "4,235"],
  [{ t: "CRBN", b: 1 }, "Low carbon (global ACWI)", "2016–2026", "2,607"],
  [{ t: "SPYX", b: 1 }, "Fossil-fuel free (S&P 500)", "2017–2026", "2,363"],
  [{ t: "ESGU", b: 1 }, "Broad ESG (ESG Aware MSCI USA)", "2018–2026", "2,107"],
  [{ t: "ESGD", b: 1 }, "International ESG (EAFE)", "2017–2026", "2,178"],
  [{ t: "NULV / ESGV / SUSL", b: 1 }, "ESG value · broad ESG · ESG leaders", "2018–2026", "1,498–2,084"],
];
grid(s, 0.85, 2.4, [2.5, 5.3, 2.0, 1.4], 0.39, panel, 12);
s.addShape(p.ShapeType.roundRect, { x: 12.0, y: 2.4, w: 0.0, h: 0.0, rectRadius: 0, fill: { color: WHITE } });
s.addText("ESG scores: Dataset/ESG_2000-2026.csv — 3,551 US companies, Refinitiv-style 0–100 ESG score (mean 40.3), E/S/G pillars, ~$83T market cap. Used to build the constructed basket.",
  { x: 0.85, y: 6.75, w: 11.4, h: 0.5, fontFace: BF, fontSize: 11.5, italic: true, color: MUTED });
notes[3] = "The panel deliberately spans ESG styles so a shared result cannot be an artifact of one construction.";

/* ------------------------------------------------------------- 4 METHOD */
s = slide(WHITE);
kicker(s, "Method", 0.85, 0.6, FOREST);
title(s, "One pipeline, applied identically to all 12");
const steps = [
  ["Features", "6 price-based, lagged z-scored:\nrv term structure, trend, realized\nvol, drawdown, momentum,\nParkinson range vol"],
  ["HMM", "3-state Gaussian, diagonal\ncovariance, Baum-Welch EM,\n8 restarts → best log-likelihood"],
  ["Labeling", "States ranked by mean realized\nvol → Calm / Choppy / Stress.\nForward-algorithm filtered decode"],
  ["Validation", "Walk-forward: monthly retrain on\nexpanding window. Every bar\nlabeled only from its own past"],
];
steps.forEach((t, i) => {
  const x = 0.85 + i * 3.0;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.0, w: 2.75, h: 2.5, rectRadius: 0.1, fill: { color: i % 2 ? "26654F" : FOREST } });
  s.addText(`${i + 1}`, { x: x + 0.18, y: 2.12, w: 0.5, h: 0.4, fontFace: HF, fontSize: 20, bold: true, color: GOLD, margin: 0 });
  s.addText(t[0], { x: x + 0.62, y: 2.14, w: 2.0, h: 0.4, fontFace: HF, fontSize: 19, bold: true, color: WHITE, margin: 0 });
  s.addText(t[1], { x: x + 0.2, y: 2.68, w: 2.4, h: 1.7, fontFace: BF, fontSize: 11.5, color: "DCE8DF", margin: 0, lineSpacing: 16 });
  if (i < 3) s.addText("→", { x: x + 2.76, y: 3.0, w: 0.26, h: 0.5, fontFace: BF, fontSize: 20, bold: true, color: MOSS, align: "center", margin: 0 });
});
s.addShape(p.ShapeType.roundRect, { x: 0.85, y: 4.85, w: 11.65, h: 1.75, rectRadius: 0.1, fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
s.addText("Guards against look-ahead", { x: 1.15, y: 5.0, w: 5, h: 0.35, fontFace: BF, fontSize: 14, bold: true, color: FOREST, margin: 0 });
[["Lagged standardization", "z-scores use mean/std through t−1 only"],
 ["Filtered posteriors", "forward algorithm — never Viterbi over the full path"],
 ["Walk-forward refits", "no bar is decoded by a model trained on it"]].forEach((r, i) => {
  const x = 1.15 + i * 3.8;
  s.addText(r[0], { x, y: 5.42, w: 3.6, h: 0.3, fontFace: BF, fontSize: 12.5, bold: true, color: INK, margin: 0, bullet: { code: "2713", indent: 12 } });
  s.addText(r[1], { x: x + 0.22, y: 5.74, w: 3.4, h: 0.6, fontFace: BF, fontSize: 11, color: "51605A", margin: 0, lineSpacing: 15 });
});
notes[4] = "Identical treatment is what makes the cross-index comparison valid.";

/* ------------------------------------------- 5 FINDING 1 — UNIVERSALITY */
s = slide(WHITE);
kicker(s, "Finding 1", 0.85, 0.6, FOREST);
title(s, "Every ESG universe separates volatility states");
s.addText("Out-of-sample stress ÷ calm volatility ratio. All 12 exceed 1.6× — the regime signal is a property of ESG equity markets, not of any one index construction.",
  { x: 0.85, y: 1.75, w: 7.3, h: 0.8, fontFace: BF, fontSize: 14, color: MUTED, lineSpacing: 20 });
s.addChart(p.ChartType.bar, [{
  name: "Stress ÷ calm vol", labels: ["ESGV", "NULV", "ESGU", "SPYX", "SUSA", "CRBN", "DSI", "ERTH", "ESGD", "esg_index", "SUSL", "ICLN"],
  values: [2.92, 2.91, 2.82, 2.71, 2.70, 2.31, 2.31, 2.16, 2.12, 2.00, 1.85, 1.68],
}], {
  x: 0.75, y: 2.7, w: 8.2, h: 4.15, barDir: "col", chartColors: [FOREST],
  showTitle: false, showLegend: false, showValue: true, dataLabelPosition: "outEnd",
  dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: '0.00"×"',
  valAxisMinVal: 0, valAxisMaxVal: 3.4, valAxisMajorUnit: 1, valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
  catAxisLabelColor: INK, catAxisLabelFontSize: 10.5, catAxisLabelFontBold: true,
  valGridLine: { color: "EAF0EB", size: 1 }, catGridLine: { style: "none" },
});
s.addShape(p.ShapeType.roundRect, { x: 9.25, y: 2.7, w: 3.25, h: 1.95, rectRadius: 0.1, fill: { color: DARK } });
s.addText("12 / 12", { x: 9.25, y: 2.85, w: 3.25, h: 0.9, fontFace: HF, fontSize: 46, bold: true, color: GOLD, align: "center", margin: 0 });
s.addText("universes show clear volatility-regime separation", { x: 9.45, y: 3.75, w: 2.85, h: 0.7, fontFace: BF, fontSize: 12.5, color: "CFE0D4", align: "center", margin: 0, lineSpacing: 16 });
s.addShape(p.ShapeType.roundRect, { x: 9.25, y: 4.85, w: 3.25, h: 2.0, rectRadius: 0.1, fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
s.addText("Holds across", { x: 9.45, y: 4.98, w: 2.85, h: 0.3, fontFace: BF, fontSize: 12, bold: true, color: FOREST, margin: 0 });
["values-screened", "low-carbon", "fossil-fuel free", "thematic", "international"].forEach((t, i) => {
  s.addText(t, { x: 9.45, y: 5.32 + i * 0.3, w: 2.85, h: 0.3, fontFace: BF, fontSize: 11.5, color: INK, bullet: { code: "2713", indent: 11 }, margin: 0 });
});
notes[5] = "The most important scientific result: regime structure is universal across ESG styles.";

/* ------------------------------- 6 FINDING 2 — THE REVERSAL */
s = slide(WHITE);
kicker(s, "Finding 2 — a result that reversed our design", 0.85, 0.6, RED);
title(s, "Published indices beat the bespoke ESG-score basket");
s.addText("We built a universe from real ESG scores expecting it to be the best regime substrate. It was the panel's worst performer on risk-adjusted improvement.",
  { x: 0.85, y: 1.78, w: 11.4, h: 0.5, fontFace: BF, fontSize: 14.5, color: MUTED });
const cmp = [
  ["", "Vol ratio (stress/calm)", "ΔSharpe (overlay − B&H)", "Drawdown cut"],
  [{ t: "Score-constructed basket", b: 1 }, "2.00", { t: "−0.51", b: 1, c: RED }, "53%"],
  [{ t: "Published ESG indices (11)", b: 1 }, { t: "2.41", b: 1, c: FOREST }, { t: "+0.04", b: 1, c: FOREST }, "54%"],
];
grid(s, 0.85, 2.5, [3.9, 2.7, 2.9, 2.15], 0.62, cmp, 13.5);
s.addText("Best individual substrates", { x: 0.85, y: 4.6, w: 5, h: 0.35, fontFace: BF, fontSize: 14, bold: true, color: INK });
const best = [
  ["Index", "B&H Sharpe", "Overlay", "ΔSharpe"],
  [{ t: "ESGU", b: 1 }, "0.69", "0.92", { t: "+0.23", b: 1, c: FOREST }],
  [{ t: "ESGV", b: 1 }, "0.68", "0.86", { t: "+0.17", b: 1, c: FOREST }],
  [{ t: "SPYX", b: 1 }, "0.74", "0.88", { t: "+0.14", b: 1, c: FOREST }],
  [{ t: "SUSA", b: 1 }, "0.52", "0.65", { t: "+0.13", b: 1, c: FOREST }],
];
grid(s, 0.85, 5.05, [1.6, 1.7, 1.4, 1.5], 0.42, best, 12);
s.addShape(p.ShapeType.roundRect, { x: 7.35, y: 4.6, w: 5.15, h: 2.3, rectRadius: 0.1, fill: { color: "FBF3E2" }, line: { color: GOLD, width: 1.5 } });
circleIcon(s, 7.65, 4.85, 0.5, GOLD, "!", DARK);
s.addText("Why it matters", { x: 8.32, y: 4.88, w: 3.9, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: "8A5A12", margin: 0, valign: "middle" });
s.addText("The bespoke ESG-score construction added complexity and cost accuracy. An off-the-shelf broad ESG index is the better regime substrate — simpler, longer history, no construction artifacts.",
  { x: 7.65, y: 5.5, w: 4.6, h: 1.2, fontFace: BF, fontSize: 13, color: "5C4A2A", margin: 0, lineSpacing: 19 });
notes[6] = "Honest negative result about our own design choice — the bespoke basket was unnecessary.";

/* -------------------------------- 7 FINDING 3 — DRAWDOWN PROTECTION */
s = slide(WHITE);
kicker(s, "Finding 3 — the most robust result", 0.85, 0.6, FOREST);
title(s, "Drawdown protection on 11 of 11 published indices");
s.addText("Maximum drawdown, buy & hold vs regime overlay (walk-forward, net of costs). Average reduction 54%.",
  { x: 0.85, y: 1.78, w: 11.4, h: 0.4, fontFace: BF, fontSize: 14, color: MUTED });
s.addChart(p.ChartType.bar, [
  { name: "Buy & Hold", labels: ["SUSA", "DSI", "ICLN", "ERTH", "NULV", "ESGU", "ESGV", "SPYX", "CRBN", "ESGD", "SUSL"],
    values: [53.9, 49.9, 72.5, 64.2, 37.0, 33.9, 33.7, 32.8, 33.1, 33.7, 27.0] },
  { name: "Regime Overlay", labels: ["SUSA", "DSI", "ICLN", "ERTH", "NULV", "ESGU", "ESGV", "SPYX", "CRBN", "ESGD", "SUSL"],
    values: [18.2, 27.8, 33.5, 42.8, 9.3, 14.3, 13.6, 18.3, 14.7, 23.1, 8.3] },
], {
  x: 0.75, y: 2.35, w: 8.5, h: 4.4, barDir: "col", chartColors: [MUTED, FOREST],
  showTitle: false, showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BF,
  showValue: false, valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20,
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFormatCode: '0"%"',
  valAxisTitle: "Max drawdown (%)", showValAxisTitle: true, valAxisTitleColor: MUTED, valAxisTitleFontSize: 11,
  catAxisLabelColor: INK, catAxisLabelFontSize: 10.5, catAxisLabelFontBold: true,
  valGridLine: { color: "EAF0EB", size: 1 }, catGridLine: { style: "none" },
});
s.addShape(p.ShapeType.roundRect, { x: 9.55, y: 2.5, w: 2.95, h: 1.9, rectRadius: 0.1, fill: { color: DARK } });
s.addText("11/11", { x: 9.55, y: 2.62, w: 2.95, h: 0.85, fontFace: HF, fontSize: 44, bold: true, color: GOLD, align: "center", margin: 0 });
s.addText("indices had drawdown reduced", { x: 9.75, y: 3.48, w: 2.55, h: 0.6, fontFace: BF, fontSize: 12.5, color: "CFE0D4", align: "center", margin: 0, lineSpacing: 16 });
s.addShape(p.ShapeType.roundRect, { x: 9.55, y: 4.6, w: 2.95, h: 1.9, rectRadius: 0.1, fill: { color: FOREST } });
s.addText("54%", { x: 9.55, y: 4.72, w: 2.95, h: 0.85, fontFace: HF, fontSize: 44, bold: true, color: WHITE, align: "center", margin: 0 });
s.addText("average drawdown cut", { x: 9.75, y: 5.58, w: 2.55, h: 0.6, fontFace: BF, fontSize: 12.5, color: "DCE8DF", align: "center", margin: 0, lineSpacing: 16 });
notes[7] = "Risk reduction is the universal, reliable benefit — more so than return enhancement.";

/* ------------------------------------- 8 FINDING 4 — DIAGNOSED DEFECT */
s = slide(WHITE);
kicker(s, "Finding 4 — root-caused", 0.85, 0.6, AMBER);
title(s, "Over-labeling Stress is what breaks the overlay");
s.addText("A clean monotone relationship: the more days a series is labeled Stress, the worse the overlay performs — it sits out too much of the bull market and dilutes the stress bucket.",
  { x: 0.85, y: 1.78, w: 11.4, h: 0.6, fontFace: BF, fontSize: 14, color: MUTED, lineSpacing: 20 });
const band = [
  ["Stress frequency", "Indices", "Mean ΔSharpe"],
  [{ t: "15–24%  (well calibrated)", b: 1, c: FOREST }, "ESGV, NULV, ESGU, SPYX, SUSA, CRBN, DSI, ERTH, ESGD", { t: "+0.04", b: 1, c: FOREST }],
  [{ t: "35–46%  (over-labeling)", b: 1, c: RED }, "esg_index (35%), SUSL (36%), ICLN (46%)", { t: "−0.14", b: 1, c: RED }],
];
grid(s, 0.85, 2.55, [3.3, 6.0, 2.35], 0.62, band, 12.5);
s.addShape(p.ShapeType.roundRect, { x: 0.85, y: 4.55, w: 5.65, h: 2.4, rectRadius: 0.1, fill: { color: DARK } });
s.addText("Root cause — a construction flaw, not a model flaw", { x: 1.15, y: 4.7, w: 5.05, h: 0.55, fontFace: BF, fontSize: 14, bold: true, color: GOLD, margin: 0, lineSpacing: 19 });
s.addText("The constructed basket has no true intraday high/low, so the builder synthesizes an OHLC bar from the cross-sectional max/min return across its 34 constituents. That is a dispersion measure, not an intraday range — it inflates the Parkinson range-vol feature and biases the model toward Stress.",
  { x: 1.15, y: 5.3, w: 5.05, h: 1.5, fontFace: BF, fontSize: 12, color: "DCE8DF", margin: 0, lineSpacing: 17 });
s.addShape(p.ShapeType.roundRect, { x: 6.85, y: 4.55, w: 5.65, h: 2.4, rectRadius: 0.1, fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
s.addText("Corroborating evidence", { x: 7.15, y: 4.7, w: 5.05, h: 0.35, fontFace: BF, fontSize: 14, bold: true, color: FOREST, margin: 0 });
s.addText("The same basket labeled only 11–13% Stress under the chronological train/test protocol — consistent with a feature-scaling artifact that the expanding walk-forward window amplifies, rather than a defect in the regime concept itself.",
  { x: 7.15, y: 5.12, w: 5.05, h: 1.3, fontFace: BF, fontSize: 12, color: "45524B", margin: 0, lineSpacing: 17 });
s.addText("Fix: use true constituent OHLC, or drop Parkinson vol for constructed baskets.",
  { x: 7.15, y: 6.45, w: 5.05, h: 0.4, fontFace: BF, fontSize: 11.5, bold: true, italic: true, color: "8A5A12", margin: 0 });
notes[8] = "We diagnosed the failure rather than reporting it as noise — the defect is in the index construction.";

/* ------------------------------- 9 FINDING 5 — NO UNIVERSAL REGIME */
s = slide(WHITE);
kicker(s, "Finding 5", 0.85, 0.6, FOREST);
title(s, "There is no single, universal “ESG regime”");
s.addText("Agreement between each index's regime path and the constructed basket's, on shared dates. Only 43–67% exact match — the market state is meaningfully universe-specific.",
  { x: 0.85, y: 1.78, w: 7.2, h: 0.7, fontFace: BF, fontSize: 14, color: MUTED, lineSpacing: 20 });
s.addChart(p.ChartType.bar, [{
  name: "Exact label match", labels: ["CRBN", "SPYX", "ESGV", "ESGU", "NULV", "ESGD", "SUSA", "DSI", "SUSL", "ERTH", "ICLN"],
  values: [67, 63, 60, 57, 55, 54, 53, 51, 50, 49, 43],
}], {
  x: 0.75, y: 2.65, w: 7.6, h: 4.1, barDir: "bar", chartColors: [FOREST],
  showTitle: false, showLegend: false, showValue: true, dataLabelPosition: "outEnd",
  dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: '0"%"',
  valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, valAxisLabelColor: MUTED,
  valAxisLabelFontSize: 10, valAxisLabelFormatCode: '0"%"',
  catAxisLabelColor: INK, catAxisLabelFontSize: 11, catAxisLabelFontBold: true,
  valGridLine: { color: "EAF0EB", size: 1 }, catGridLine: { style: "none" },
});
s.addShape(p.ShapeType.roundRect, { x: 8.65, y: 2.65, w: 3.85, h: 2.0, rectRadius: 0.1, fill: { color: DARK } });
s.addText("43–67%", { x: 8.65, y: 2.8, w: 3.85, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: GOLD, align: "center", margin: 0 });
s.addText("label agreement across ESG universes", { x: 8.9, y: 3.7, w: 3.35, h: 0.7, fontFace: BF, fontSize: 12.5, color: "CFE0D4", align: "center", margin: 0, lineSpacing: 16 });
s.addShape(p.ShapeType.roundRect, { x: 8.65, y: 4.85, w: 3.85, h: 1.9, rectRadius: 0.1, fill: { color: CARD }, line: { color: "E1E9E3", width: 1 } });
s.addText("Implication", { x: 8.9, y: 5.0, w: 3.35, h: 0.3, fontFace: BF, fontSize: 13, bold: true, color: FOREST, margin: 0 });
s.addText("Detect regimes on the same universe the allocator actually trades. A regime path borrowed from another ESG index will disagree roughly half the time.",
  { x: 8.9, y: 5.35, w: 3.35, h: 1.3, fontFace: BF, fontSize: 12, color: "45524B", margin: 0, lineSpacing: 17 });
notes[9] = "Regimes are universe-specific — an important design constraint for the RL integration.";

/* ------------------------------------------- 10 INTEGRATION / NEXT */
s = slide(DARK);
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.16, fill: { color: GOLD } });
kicker(s, "Recommendations & integration", 0.9, 0.72, MOSS);
s.addText("Wiring the regime signal into ESG-Adaptive RL", { x: 0.85, y: 1.1, w: 11.6, h: 0.8,
  fontFace: HF, fontSize: 31, bold: true, color: WHITE });
const recs = [
  ["1", "Switch the regime substrate to a broad ESG index", "ESGU / ESGV / SUSA — best separation (2.7–2.9×) and best ΔSharpe. SUSA adds 20 years of history."],
  ["2", "Fix or retire the synthetic-OHLC basket", "Use true constituent OHLC, or drop Parkinson vol for constructed universes."],
  ["3", "Align regime universe with the trading universe", "Label agreement is only 43–67% across indices — don't borrow a regime path."],
];
recs.forEach((r, i) => {
  const y = 2.2 + i * 1.15;
  s.addShape(p.ShapeType.roundRect, { x: 0.85, y, w: 7.2, h: 1.0, rectRadius: 0.08, fill: { color: "1D453A" } });
  circleIcon(s, 1.12, y + 0.24, 0.52, GOLD, r[0], DARK);
  s.addText(r[1], { x: 1.85, y: y + 0.11, w: 6.0, h: 0.4, fontFace: BF, fontSize: 14.5, bold: true, color: WHITE, margin: 0 });
  s.addText(r[2], { x: 1.85, y: y + 0.48, w: 6.05, h: 0.5, fontFace: BF, fontSize: 11.5, color: "AECBB7", margin: 0, lineSpacing: 15 });
});
s.addShape(p.ShapeType.roundRect, { x: 8.4, y: 2.2, w: 4.1, h: 3.45, rectRadius: 0.1, fill: { color: "10281F" }, line: { color: FOREST, width: 1 } });
s.addText("Integration path", { x: 8.7, y: 2.4, w: 3.5, h: 0.45, fontFace: HF, fontSize: 19, bold: true, color: GOLD, margin: 0 });
[
  "esg_regime.label_regimes() emits a daily Calm/Choppy/Stress label per bar",
  "PortfolioEnv exposes the label as an observation feature",
  "RewardWeights become per-regime: the evolutionary layer searches one weight vector per state",
  "Evaluate the price of virtue conditional on regime",
].forEach((t, i) => {
  s.addText(t, { x: 8.7, y: 2.92 + i * 0.68, w: 3.55, h: 0.68, fontFace: BF, fontSize: 11.5,
    color: "DCE8DF", bullet: { code: "2192", indent: 15 }, margin: 0, lineSpacing: 15 });
});
s.addText("All code, data, results and this deck committed on branch esg-regime-hmm · ESG-Adaptive RL · targeting ACM ICAIF 2026",
  { x: 0.85, y: 6.6, w: 11.6, h: 0.4, fontFace: BF, fontSize: 12, italic: true, color: "8FB39B", align: "center" });
notes[10] = "Concrete next actions and how the regime layer plugs into the RL environment and reward.";

p.slides.forEach((sl, i) => { if (notes[i + 1]) sl.addNotes(notes[i + 1]); });
p.writeFile({ fileName: "ESG_Regime_Findings.pptx" }).then(f => console.log("WROTE", f));
