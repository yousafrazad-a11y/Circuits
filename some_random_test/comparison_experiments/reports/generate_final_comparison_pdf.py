"""Generate the final IOI pruning comparison PDF from saved JSON metrics."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).with_name("final_ioi_circuit_comparison.pdf")
PAGE_W, PAGE_H = 842, 595  # A4 landscape, points


def load(name):
    return json.loads((RESULTS / name).read_text())


G_KL = load("global_node_pressure8x_epoch_0080.metrics.json")
G_EPOCH = load("global_node_pressure8x_final.metrics.json")
G = G_KL  # Legacy detailed layout below uses the KL-matched normal checkpoint.
P = load("position_node_pressure8x_final.metrics.json")
E5 = load("unified_peap.metrics.json")
E10 = load("unified_peap_topk_10000.metrics.json")
SEL = load("node_kl_matched_selection.json")


def esc(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PDF:
    def __init__(self):
        self.pages = []

    def page(self):
        self.ops = []
        self.pages.append(self.ops)

    def text(self, x, y, value, size=9, bold=False, color=(0.12, 0.15, 0.20)):
        value = str(value).encode("ascii", "replace").decode("ascii")
        font = "F2" if bold else "F1"
        r, g, b = color
        self.ops.append(f"BT /{font} {size} Tf {r} {g} {b} rg {x} {y} Td ({esc(value)}) Tj ET")

    def line(self, x1, y1, x2, y2, color=(0.75, 0.78, 0.82), width=.5):
        r, g, b = color
        self.ops.append(f"{r} {g} {b} RG {width} w {x1} {y1} m {x2} {y2} l S")

    def fill(self, x, y, w, h, color):
        r, g, b = color
        self.ops.append(f"{r} {g} {b} rg {x} {y} {w} {h} re f")

    def paragraph(self, x, y, value, width=110, size=9, leading=12, bold=False):
        for line in wrap(value, width=width, break_long_words=False):
            self.text(x, y, line, size, bold)
            y -= leading
        return y

    def table(self, x, y, headers, rows, widths, size=7.5, row_h=20):
        total = sum(widths)
        self.fill(x, y-row_h, total, row_h, (0.10, 0.27, 0.43))
        cx = x
        for head, width in zip(headers, widths):
            self.text(cx+4, y-13, head, size, True, (1, 1, 1)); cx += width
        y -= row_h
        for i, row in enumerate(rows):
            if i % 2 == 0:
                self.fill(x, y-row_h, total, row_h, (0.94, 0.96, 0.98))
            cx = x
            for value, width in zip(row, widths):
                self.text(cx+4, y-13, value, size); cx += width
            self.line(x, y-row_h, x+total, y-row_h)
            y -= row_h
        return y

    def header(self, title, subtitle):
        self.fill(0, PAGE_H-76, PAGE_W, 76, (0.06, 0.20, 0.33))
        self.text(34, PAGE_H-36, title, 20, True, (1, 1, 1))
        self.text(34, PAGE_H-57, subtitle, 9, False, (.82, .90, .96))

    def footer(self, number):
        self.line(32, 25, PAGE_W-32, 25)
        self.text(34, 12, "Generated from saved experiment JSON; GPT-2 small, IOI ABBA.", 7)
        self.text(PAGE_W-55, 12, str(number), 7)

    def save(self, path):
        objects = [None, None, None, None]  # catalog, pages, two fonts
        objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        page_ids = []
        for ops in self.pages:
            stream = ("\n".join(ops) + "\n").encode("latin-1")
            content_id = len(objects) + 1
            objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream")
            page_id = len(objects) + 1
            objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>".encode())
            page_ids.append(page_id)
        objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>".encode()
        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, 1):
            offsets.append(len(data)); data.extend(f"{i} 0 obj\n".encode()+obj+b"\nendobj\n")
        xref = len(data); data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
        for offset in offsets[1:]: data.extend(f"{offset:010d} 00000 n \n".encode())
        data.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        path.write_bytes(data)


def pct(x): return f"{100*x:.1f}%"
def f3(x): return f"{x:.3f}"
def count(x): return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


pdf = PDF()

# Concise two-page comparison requested for the final handoff.  The older
# detailed layout remains below as reference but is intentionally unreachable.
def generate_clear_report():
    # Page 1: the two node-pruning methods.
    pdf.page(); pdf.header("Node-Pruning Comparison", "Normal node pruning at two operating points versus the final position-aware checkpoint")
    y = 492
    pdf.text(34, y, "Circuit sizes", 13, True); y -= 12
    pc = P["structure"]["effective_gate_slots_by_granularity"]
    pe = P["structure"]["high_level_edge_pruning"]["all_high_level_edges"]
    circuits = [G_KL, G_EPOCH, P]

    def size_pair(item):
        return "{}/{}".format(count(item["active"]), count(item["total"])), "{:.1f}%".format(item["percent_pruned"])

    size_rows = []
    edge_items = [d["structure"]["high_level_edge_pruning"]["all_high_level_edges"] for d in circuits]
    size_rows.append(["High-level abstract edges", *sum((size_pair(x) for x in edge_items), ())])
    for key, label in [
        ("attention_blocks", "Attention blocks"),
        ("attention_heads", "Attention heads"),
        ("attention_neurons", "Attention head neurons"),
        ("mlp_blocks", "MLP blocks"),
        ("mlp_hidden", "MLP hidden neurons"),
        ("mlp_output", "MLP output neurons"),
    ]:
        items = [d["structure"]["effective_gate_slots_by_granularity"][key] for d in circuits]
        size_rows.append([label, *sum((size_pair(x) for x in items), ())])
    parameter_items = [d["structure"]["parameter_pruning_proxy"]["whole_model_effective_parameters"] for d in circuits]
    size_rows.append(["Whole-model parameter proxy", *sum((size_pair(x) for x in parameter_items), ())])
    y = pdf.table(34, y, ["Granularity / size measure", "KL-match open/total", "KL closed", "Epoch-match open/total", "Epoch closed", "Position open/total", "Position closed"], size_rows, [155,120,76,125,76,125,76], 6.6, 21) - 15
    pdf.text(34, y, "Performance on the shared 500-example test set", 13, True); y -= 12
    metric_rows = []
    for label, d in [("Normal - KL matched (ep. 80)", G_KL), ("Normal - epoch matched (final)", G_EPOCH), ("Position-aware - final", P)]:
        metric_rows.append([label, pct(d["accuracy"]), pct(d["exact_match"]), pct(d["pairwise_accuracy"]), f3(d["kl_div"]), f3(d["logit_diff"]), f3(d["soft_faithfulness"])])
    pdf.table(34, y, ["Circuit", "Accuracy", "Exact", "Pairwise", "KL", "Logit diff", "Faithfulness"], metric_rows, [205,85,80,85,80,95,100], 8, 26)
    pdf.text(34, 49, "Open/total counts are hierarchy-finalized effective slots; closed columns are percentages. Position totals include nine logical sections.", 8)
    pdf.footer(1)

    # Page 2: direct position-aware node versus PEAP comparison.
    pdf.page(); pdf.header("Position-Aware Node Pruning vs PEAP", "Final position-aware node checkpoint; two retained PEAP operating points")
    y = 492
    pdf.text(34, y, "Circuit sizes", 13, True); y -= 12
    e5a = E5["structure"]["argo_style_peap_compatible_comparison"]["selected_edges_against_full_graph"]
    e10a = E10["structure"]["argo_style_peap_compatible_comparison"]["selected_edges_against_full_graph"]
    e5c = E5["structure"]["concrete_token_level_edges"]
    e10c = E10["structure"]["concrete_token_level_edges"]
    pnodes = pc["attention_heads"]["active"] + pc["mlp_blocks"]["active"]
    pnodes_full = pc["attention_heads"]["total"] + pc["mlp_blocks"]["total"]
    s5 = E5["structure"]["selected_abstract_graph"]
    s10 = E10["structure"]["selected_abstract_graph"]
    size_rows = [
        ["Position-aware node", f'{count(pe["active"])}/{count(pe["total"])}', f'{pe["percent_pruned"]:.2f}%', f'{pnodes}/{pnodes_full} component slots', "--"],
        ["PEAP top-5k", f'{count(e5a["active"])}/{count(e5a["total"])}', f'{e5a["percent_pruned"]:.2f}%', f'{s5["nodes"]} selected nodes', f'{count(e5c["active"])}/{count(e5c["total"])}'],
        ["PEAP top-10k", f'{count(e10a["active"])}/{count(e10a["total"])}', f'{e10a["percent_pruned"]:.2f}%', f'{s10["nodes"]} selected nodes', f'{count(e10c["active"])}/{count(e10c["total"])}'],
    ]
    y = pdf.table(34, y, ["Circuit", "Abstract edges active/full", "Edges closed", "High-level nodes", "Native PEAP token edges"], size_rows, [160,185,105,170,160], 8, 27) - 25
    pdf.text(34, y, "Performance on the shared 500-example test set", 13, True); y -= 12
    metric_rows = []
    for label, d in [("Position-aware node", P), ("PEAP top-5k", E5), ("PEAP top-10k", E10)]:
        metric_rows.append([label, pct(d["accuracy"]), pct(d["exact_match"]), pct(d["pairwise_accuracy"]), f3(d["kl_div"]), f3(d["logit_diff"]), f3(d["soft_faithfulness"])])
    pdf.table(34, y, ["Circuit", "Accuracy", "Exact", "Pairwise", "KL", "Logit diff", "Faithfulness"], metric_rows, [205,85,80,85,80,95,100], 8, 26)
    pdf.paragraph(34, 88, "Size rule: surviving nodes imply every valid incident edge, following circuit_pruning-argo. For PEAP compatibility, Q/K/V are collapsed into one abstract connection and causal logical-position transport is included.", 130, 8, 10)
    pdf.text(34, 43, "KL is D_KL(full || circuit); logit difference is IO minus distractor; faithfulness is circuit/full logit difference.", 8)
    pdf.footer(2)
    pdf.save(OUT)
    print(OUT)


generate_clear_report()
raise SystemExit(0)

# Page 1: evaluation outcomes
pdf.page(); pdf.header("IOI Circuit Pruning: Final Comparative Results", "Four selected circuits; one shared 500-example held-out ABBA evaluation")
y = 492
pdf.text(34, y, "Evaluation performance", 13, True); y -= 12
rows = []
for label, d in [("Node (global), ep.500", G), ("Node (position), ep.230", P), ("PEAP, top-5k", E5), ("PEAP, top-10k", E10)]:
    rows.append([label, str(d["n_examples"]), pct(d["accuracy"]), pct(d["exact_match"]), pct(d["pairwise_accuracy"]), f3(d["kl_div"]), f3(d["logit_diff"]), f3(d["full_model_logit_diff"]), f3(d["soft_faithfulness"])])
y = pdf.table(34, y, ["Circuit", "N", "Acc.", "Exact", "Pairwise", "KL", "Logit diff", "Full diff", "Faith."], rows, [151,32,52,52,57,55,70,64,55], 7.2, 23)-18
pdf.text(34, y, "Interpretation", 12, True); y -= 17
y = pdf.paragraph(34, y, "The KL-matched node pair has held-out KL 0.1185 (global) versus 0.1037 (position-aware), with accuracies 96.4% and 97.6%. PEAP top-5k is the approximately KL=0.1 operating point (KL 0.1000); PEAP top-10k is the accuracy-oriented point (97.4%, KL 0.0291).", 130, 9, 12)-6
y = pdf.paragraph(34, y, "Metric definitions: KL is D_KL(full || circuit) over the complete next-token vocabulary. Accuracy is exact correct-token accuracy. Pairwise accuracy tests whether IO logit >= distractor logit. Exact is circuit/full-model argmax agreement. Logit difference is IO minus distractor; soft faithfulness is circuit logit difference divided by full-model logit difference.", 130, 8.5, 11)-8
pdf.text(34, y, "Selection protocol", 12, True); y -= 17
selrows = [["Global node, ep.500", f3(SEL["global_node"]["validation_kl_div"]), pct(SEL["global_node"]["validation_accuracy"]), "fixed final"], ["Position node, ep.230", f3(SEL["position_node"]["validation_kl_div"]), pct(SEL["position_node"]["validation_accuracy"]), "closest validation KL"], ["PEAP top-5k", "0.089", "93.0%", "KL~0.1 point"], ["PEAP top-10k", "0.028", "98.0%", "accuracy point"]]
pdf.table(34, y, ["Selected circuit", "Monitor/val. KL", "Monitor/val. acc.", "Reason"], selrows, [190,110,120,190], 8, 21)
pdf.footer(1)

# Page 2: size overview
pdf.page(); pdf.header("Direct Structural Comparison", "Position-aware node pruning and PEAP on one argo-derived abstract edge denominator")
y=492
pdf.text(34,y,"Shared position-aware abstract graph",13,True); y-=12
ge=G["structure"]["high_level_edge_pruning"]["all_high_level_edges"]; pe=P["structure"]["high_level_edge_pruning"]["all_high_level_edges"]
e5=E5["structure"]["concrete_token_level_edges"]; e10=E10["structure"]["concrete_token_level_edges"]
e5a=E5["structure"]["argo_style_peap_compatible_comparison"]["selected_edges_against_full_graph"]
e10a=E10["structure"]["argo_style_peap_compatible_comparison"]["selected_edges_against_full_graph"]
rows=[["Position-aware node",count(pe["active"]),count(pe["closed"]),count(pe["total"]),f'{pe["percent_pruned"]:.2f}%'],
      ["PEAP top-5k",count(e5a["active"]),count(e5a["closed"]),count(e5a["total"]),f'{e5a["percent_pruned"]:.2f}%'],
      ["PEAP top-10k",count(e10a["active"]),count(e10a["closed"]),count(e10a["total"]),f'{e10a["percent_pruned"]:.2f}%']]
y=pdf.table(34,y,["Circuit","Active abstract edges","Closed abstract edges","Full graph","Closed (%)"],rows,[190,150,155,125,110],8,25)-17
pdf.text(34,y,"Position-aware edge categories",12,True); y-=12
poscross=P["structure"]["high_level_edge_pruning"]["cross_position_attention_edges"]
rows=[["Node: all",count(pe["active"]),count(pe["closed"]),count(pe["total"]),f'{pe["percent_pruned"]:.2f}%'],
      ["Position: cross-position attention",count(poscross["active"]),count(poscross["closed"]),count(poscross["total"]),f'{poscross["percent_pruned"]:.2f}%']]
y=pdf.table(34,y,["Edge set","Active","Closed","Full","Closed (%)"],rows,[245,100,100,100,100],8,22)-17
pdf.text(34,y,"Native PEAP concrete token-edge counts",12,True); y-=12
rows=[["PEAP top-5k",f'{count(e5["active"])}/{count(e5["total"])}',f'{e5["percent_pruned"]:.2f}%',str(E5["structure"]["selected_abstract_graph"]["nodes"]),str(E5["structure"]["selected_abstract_graph"]["edges"])],
      ["PEAP top-10k",f'{count(e10["active"])}/{count(e10["total"])}',f'{e10["percent_pruned"]:.2f}%',str(E10["structure"]["selected_abstract_graph"]["nodes"]),str(E10["structure"]["selected_abstract_graph"]["edges"])]]
y=pdf.table(34,y,["Circuit","Mean active/full token edges","Closed","Selected nodes","Selected edges"],rows,[145,220,100,110,110],8,22)-15
pdf.text(34,y,"Position-aware node extraction proxy",12,True); y-=12
rows=[]
for label,d in [("Node (position)",P)]:
    pp=d["structure"]["parameter_pruning_proxy"]
    rows.append([label,f'{pp["prunable_parameters"]["percent_pruned"]:.2f}%',f'{pp["whole_model_effective_parameters"]["percent_pruned"]:.2f}%',count(pp["whole_model_effective_parameters"]["active"]),count(pp["whole_model_effective_parameters"]["total"])])
y=pdf.table(34,y,["Circuit","Prunable params closed","Whole-model proxy closed","Effective params","Base params"],rows,[150,145,155,120,120],8,23)-14
pdf.paragraph(34,y,"Shared counting follows circuit_pruning-argo's dense-survivor rule: surviving nodes imply every valid incident edge. PEAP adaptations collapse Q/K/V into one abstract source-to-head connection and add causal logical-section attention transport. The 109,722-edge denominator is therefore identical for both methods. Native concrete token-edge counts remain separate because they vary with prompt length. Physical parameter extraction is also distinct: the position-aware node union across sections closes only 1.15% of the whole-model proxy.",130,8.5,11)
pdf.footer(2)

# Page 3: node granularity and sections
pdf.page(); pdf.header("Node-Pruning Granularity", "Hierarchy-finalized effective gates; parent closure propagates to all children")
y=492
cats=[("attention_blocks","Attention blocks"),("attention_heads","Attention heads"),("attention_neurons","Attention neurons"),("mlp_blocks","MLP blocks"),("mlp_hidden","MLP hidden neurons"),("mlp_output","MLP output dims")]
rows=[]
for key,label in cats:
    a=G["structure"]["effective_gate_slots_by_granularity"][key]; b=P["structure"]["effective_gate_slots_by_granularity"][key]
    rows.append([label,f'{count(a["active"])}/{count(a["total"])}',f'{a["percent_pruned"]:.2f}%',f'{count(b["active"])}/{count(b["total"])}',f'{b["percent_pruned"]:.2f}%'])
y=pdf.table(34,y,["Granularity","Global active/full","Global closed","Position active/full","Position closed"],rows,[210,145,105,150,105],8,23)-18
pdf.text(34,y,"Position-aware high-level circuit by semantic section",12,True); y-=12
order=["prefix","IO","and","S1","S1+1","action1","S2","action2","to"]
sec=P["structure"]["effective_gate_slots_by_section"]; edge_sec=P["structure"]["high_level_edge_pruning"]["by_source_section_for_non_crossing_edges"]
rows=[]
for name in order:
    h=sec[name]["attention_heads"]; m=sec[name]["mlp_blocks"]; e=edge_sec[name]
    rows.append([name,f'{h["active"]}/{h["total"]}',f'{h["percent_pruned"]:.1f}%',f'{m["active"]}/{m["total"]}',f'{m["percent_pruned"]:.1f}%',f'{e["percent_pruned"]:.1f}%'])
pdf.table(34,y,["Section","Heads active","Heads closed","MLPs active","MLPs closed","Local edges closed"],rows,[125,115,115,115,115,130],7.8,20)
pdf.text(475,47,"Nine semantic sections; variable token lengths map into these fixed logical roles.",7.5)
pdf.footer(3)

# Page 4: PEAP structure, assumptions, provenance
pdf.page(); pdf.header("PEAP Structure and Comparison Notes", "Two retained operating points from one position-aware edge-attribution ranking")
y=492
pdf.text(34,y,"Selected abstract PEAP graphs",13,True); y-=12
def peap_rows(d,label):
    s=d["structure"]["selected_abstract_graph"]; nt=s["nodes_by_type"]; rel=s["edges_by_position_relation"]
    return [label,str(s["nodes"]),str(nt.get("h",0)),str(nt.get("m",0)),str(nt.get("r",0)),str(s["edges"]),str(rel.get("crossing_attention",0)),str(rel.get("same_position_or_component",0))]
y=pdf.table(34,y,["Circuit","Nodes","Head nodes","MLP nodes","Residual","Edges","Cross-pos.","Other edges"],[peap_rows(E5,"PEAP top-5k"),peap_rows(E10,"PEAP top-10k")],[120,70,90,80,70,80,90,100],8,24)-18
pdf.text(34,y,"Edges by upstream -> downstream type",12,True); y-=12
types=["h_to_h","h_to_m","h_to_r","m_to_h","m_to_m","m_to_r","r_to_h","r_to_m"]
rows=[]
for label,d in [("top-5k",E5),("top-10k",E10)]:
    et=d["structure"]["selected_abstract_graph"]["edges_by_upstream_to_downstream_type"]
    rows.append([label]+[str(et.get(k,0)) for k in types])
y=pdf.table(34,y,["Circuit","h->h","h->m","h->r","m->h","m->m","m->r","r->h","r->m"],rows,[105]+[75]*8,7.5,23)-18
pdf.text(34,y,"Experimental controls and interpretation",12,True); y-=16
notes=[
"All reported performance rows use the same ordered 500-example held-out IOI ABBA set and the same shared vocabulary-level metric implementation. GPT-2 small is the reference full model.",
"Node checkpoints are hardened and hierarchy-finalized before inference. Global pruning has one gate row; position-aware pruning has nine logical-section rows. Low-level neuron/dimension pruning exists only for the node methods.",
"For node high-level edge accounting, heads and MLP blocks are nodes, Q/K/V are collapsed to one abstract connection, residual endpoints remain open, and an edge closes when either gated endpoint closes. Position-aware counting includes causal section-pair attention transport; only the final section connects to next-token output.",
"PEAP natively selects directed edges. Its abstract graph size is fixed by the selected circuit, while concrete token-edge counts vary by prompt length and are therefore reported as held-out means. No head/MLP-neuron percentage is claimed for PEAP.",
"The test set was used only after checkpoint/operating-point selection. The node pair was matched using the 100-example validation view (absolute validation-KL gap 0.000052). Two PEAP points are retained because matching KL and accuracy simultaneously was not possible."
]
for n in notes:
    y=pdf.paragraph(44,y,"- "+n,123,8.4,11)-5
pdf.footer(4)

pdf.save(OUT)
print(OUT)
